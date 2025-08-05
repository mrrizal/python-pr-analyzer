from io import StringIO
import os
import json
import asyncio
import requests
import aiohttp
import ast
import asttokens
from dotenv import load_dotenv
from unidiff import PatchSet
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class FileData:
    """Data class representing a file in the PR"""
    filename: str
    patch: str
    sha: str


@dataclass
class AddedLine:
    """Data class representing an added line with its line number"""
    line_number: int
    content: str


@dataclass
class DeletedLine:
    """Data class representing a deleted line with its original line number"""
    original_line_number: int
    content: str


@dataclass
class FunctionChange:
    """Data class representing a function with its added and deleted lines"""
    function_name: str
    function_code: str
    function_start_line: int
    function_end_line: int
    added_lines: List[AddedLine]
    deleted_lines: List[DeletedLine]  # New field for deleted lines
    sha: str

    @property
    def has_changes(self) -> bool:
        """Check if function has any meaningful changes"""
        return (len(self.added_lines) > 0 or len(self.deleted_lines) > 0) and not (
            len(self.added_lines) == 1 and
            len(self.deleted_lines) == 0 and
            not self.added_lines[0].content.strip()
        )


class GitHubAPIError(Exception):
    """Custom exception for GitHub API related errors"""
    pass


class CodeAnalysisError(Exception):
    """Custom exception for code analysis related errors"""
    pass


class GitHubClient:
    """GitHub API client implementation"""

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_headers = {
            "Authorization": f"token {token}" if token else None
        }

    def _make_request(self, url: str, headers: Dict[str, str]) -> requests.Response:
        """Make HTTP request with proper error handling"""
        response = requests.get(url, headers=headers)
        if response.status_code == 404:
            raise GitHubAPIError(f"Resource not found: {url}")
        elif response.status_code == 403:
            raise GitHubAPIError("GitHub API rate limit exceeded or access denied")
        elif response.status_code >= 400:
            raise GitHubAPIError(f"GitHub API error {response.status_code}: {response.text}")
        return response

    def get_pr_diff(self, repo: str, pr_number: int) -> str:
        """Download PR diff from GitHub API"""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        headers = self.base_headers.copy()
        headers["Accept"] = "application/vnd.github.v3.diff"

        response = self._make_request(url, headers)
        return response.text

    def get_pr_info(self, repo: str, pr_number: int) -> Dict:
        """Get PR information including commit SHA"""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        headers = self.base_headers.copy()
        headers["Accept"] = "application/vnd.github.v3+json"

        response = self._make_request(url, headers)
        return response.json()

    def get_raw_file(self, repo: str, commit_sha: str, filepath: str) -> str:
        """Download raw file content from GitHub (after changes)"""
        url = f"https://raw.githubusercontent.com/{repo}/{commit_sha}/{filepath}"
        headers = {"Accept": "application/vnd.github.v3.raw"}

        response = self._make_request(url, headers)
        return response.text

    def get_raw_file_before_changes(self, repo: str, commit_sha: str, filepath: str) -> str:
        """Download raw file content before changes (base commit)"""
        # For simplicity, we'll use the parent commit. In a real implementation,
        # you might want to get the actual base SHA from the PR info
        url = f"https://raw.githubusercontent.com/{repo}/{commit_sha}~1/{filepath}"
        headers = {"Accept": "application/vnd.github.v3.raw"}

        response = self._make_request(url, headers)
        return response.text


class JSONFileStorage:
    """JSON file storage implementation"""

    def save_files_data(self, data: List[FileData], filepath: str) -> None:
        """Save files data to JSON file"""
        json_data = [
            {
                "filename": file_data.filename,
                "patch": file_data.patch,
                "sha": file_data.sha
            }
            for file_data in data
        ]
        Path(filepath).write_text(json.dumps(json_data, indent=2))

    def load_files_data(self, filepath: str) -> List[FileData]:
        """Load files data from JSON file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Files data not found: {filepath}")

        with open(filepath, "r") as f:
            json_data = json.load(f)

        return [
            FileData(
                filename=item["filename"],
                patch=item["patch"],
                sha=item["sha"]
            )
            for item in json_data
        ]


class UnidiffPatchParser:
    """Patch parser using unidiff library"""

    def parse_diff(self, diff_text: str, head_sha: str) -> List[FileData]:
        """Parse diff text into FileData objects"""
        patch_set = PatchSet(StringIO(diff_text))

        files_data = []
        for patch in patch_set:
            filename = self._normalize_filename(patch.source_file)
            files_data.append(FileData(
                filename=filename,
                patch=str(patch),
                sha=head_sha
            ))

        return files_data

    def _normalize_filename(self, filename: str) -> str:
        """Normalize filename by removing git prefixes"""
        if filename.startswith("a/"):
            return filename[2:]
        return filename

    def extract_added_lines(self, patch: str) -> List[AddedLine]:
        """Extract added lines from patch with line numbers"""
        removed_lines_set = self._get_removed_lines(patch)
        added_lines = []
        line_number = None

        for line in patch.splitlines():
            if line.startswith("@@"):
                line_number = self._extract_start_line(line)
            elif self._is_added_line(line) and line_number is not None:
                raw_content = line[1:]
                if raw_content.strip() not in removed_lines_set:
                    line_number += 1
                    added_lines.append(AddedLine(line_number, raw_content))
            elif not line.startswith("-") and line_number is not None:
                line_number += 1

        return added_lines

    def extract_deleted_lines(self, patch: str) -> List[DeletedLine]:
        """Extract deleted lines from patch with original line numbers"""
        deleted_lines = []
        original_line_number = None

        for line in patch.splitlines():
            if line.startswith("@@"):
                original_line_number = self._extract_original_start_line(line)
            elif self._is_deleted_line(line) and original_line_number is not None:
                raw_content = line[1:]
                deleted_lines.append(DeletedLine(original_line_number, raw_content))
                original_line_number += 1
            elif not line.startswith("+") and original_line_number is not None:
                original_line_number += 1

        return deleted_lines

    def _get_removed_lines(self, patch: str) -> set:
        """Extract all removed lines to filter out moved code"""
        return {
            line[1:].strip() for line in patch.splitlines()
            if line.startswith("-") and not line.startswith("---")
        }

    def _extract_start_line(self, hunk_header: str) -> int:
        """Extract starting line number from hunk header (for new file)"""
        parts = hunk_header.split(" ")
        for part in parts:
            if part.startswith("+"):
                return int(part[1:].split(",")[0]) - 1
        return 0

    def _extract_original_start_line(self, hunk_header: str) -> int:
        """Extract starting line number from hunk header (for original file)"""
        parts = hunk_header.split(" ")
        for part in parts:
            if part.startswith("-"):
                return int(part[1:].split(",")[0])
        return 1

    def _is_added_line(self, line: str) -> bool:
        """Check if line is an added line (not header)"""
        return (line.startswith("+") and
                not line.startswith("+++") and
                not line.startswith("@@"))

    def _is_deleted_line(self, line: str) -> bool:
        """Check if line is a deleted line (not header)"""
        return (line.startswith("-") and
                not line.startswith("---") and
                not line.startswith("@@"))


class ASTCodeAnalyzer:
    """AST-based code analyzer"""

    def get_enclosing_node(self, source_code: str, target_line: int) -> Optional[ast.AST]:
        """Find the smallest enclosing function/class node for a given line"""
        try:
            atok = asttokens.ASTTokens(source_code, parse=True)
            root = atok.tree
            candidates = []

            for node in ast.walk(root):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if (hasattr(node, 'lineno') and hasattr(node, 'end_lineno') and
                        node.lineno <= target_line <= node.end_lineno):
                        candidates.append((node.lineno, node.end_lineno, node))

            if candidates:
                # Return the smallest enclosing node
                _, _, best_node = min(candidates, key=lambda x: x[1] - x[0])
                return best_node

        except SyntaxError:
            raise CodeAnalysisError(f"Syntax error in source code at line {target_line}")

        return None

    def extract_node_source(self, source_code: str, node: ast.AST) -> Optional[str]:
        """Extract source code for a given AST node"""
        return ast.get_source_segment(source_code, node)

    def get_function_name(self, node: ast.AST) -> str:
        """Get the name of a function/class node"""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        return "unknown"


class PRDiffAnalyzer:
    """Main class for analyzing PR diffs following SRP"""

    def __init__(
        self,
        github_client: GitHubClient,
        file_storage: JSONFileStorage,
        patch_parser: UnidiffPatchParser,
        code_analyzer: ASTCodeAnalyzer
    ):
        self.github_client = github_client
        self.file_storage = file_storage
        self.patch_parser = patch_parser
        self.code_analyzer = code_analyzer

    def download_and_save_pr_data(self, repo: str, pr_number: int, output_file: str) -> None:
        """Download PR diff and save to file"""
        diff_text = self.github_client.get_pr_diff(repo, pr_number)
        pr_info = self.github_client.get_pr_info(repo, pr_number)
        head_sha = pr_info['head']['sha']

        files_data = self.patch_parser.parse_diff(diff_text, head_sha)
        self.file_storage.save_files_data(files_data, output_file)

    def analyze_pr_changes(self, files_data_path: str, repo: str) -> Dict[str, List[FunctionChange]]:
        """Analyze changes in PR and return dict with functions and their added/deleted lines"""
        files_data = self.file_storage.load_files_data(files_data_path)
        result = {}

        for file_data in files_data:
            added_lines = self.patch_parser.extract_added_lines(file_data.patch)
            deleted_lines = self.patch_parser.extract_deleted_lines(file_data.patch)

            if not file_data.filename.endswith(".py"):
                continue

            if not added_lines and not deleted_lines:
                continue

            try:
                # Get current version of the file (after changes)
                current_source_code = self.github_client.get_raw_file(repo, file_data.sha, file_data.filename)

                # Get previous version of the file (before changes) for deleted line analysis
                previous_source_code = None
                try:
                    previous_source_code = self.github_client.get_raw_file_before_changes(repo, file_data.sha, file_data.filename)
                except GitHubAPIError:
                    # File might be new, so no previous version exists
                    pass

                # Get functions with their added and deleted lines
                function_changes = self._get_functions_with_changes(
                    current_source_code, previous_source_code, added_lines, deleted_lines, file_data.sha
                )

                # Add to result dict
                result[file_data.filename] = function_changes

            except GitHubAPIError as e:
                print(f"Error fetching file content for {file_data.filename}: {e}")
                continue

        return result

    def _get_functions_with_changes(
        self,
        current_source_code: str,
        previous_source_code: Optional[str],
        added_lines: List[AddedLine],
        deleted_lines: List[DeletedLine],
        sha: str
    ) -> List[FunctionChange]:
        """Get functions that contain added or deleted lines with mapping"""
        function_changes = {}

        # Process added lines using current source code
        for added_line in added_lines:
            try:
                node = self.code_analyzer.get_enclosing_node(current_source_code, added_line.line_number)
            except CodeAnalysisError:
                continue

            if not node:
                continue

            function_name = self.code_analyzer.get_function_name(node)
            function_code = self.code_analyzer.extract_node_source(current_source_code, node)

            if not function_code:
                continue

            # Create a unique key for each function
            func_key = f"{function_name}_{node.lineno}"

            if func_key not in function_changes:
                function_changes[func_key] = FunctionChange(
                    function_name=function_name,
                    function_code=function_code,
                    function_start_line=node.lineno,
                    function_end_line=node.end_lineno,
                    added_lines=[],
                    deleted_lines=[],
                    sha=sha
                )

            function_changes[func_key].added_lines.append(added_line)

        # Process deleted lines using previous source code (if available)
        if previous_source_code and deleted_lines:
            for deleted_line in deleted_lines:
                try:
                    node = self.code_analyzer.get_enclosing_node(previous_source_code, deleted_line.original_line_number)
                except CodeAnalysisError:
                    continue

                if not node:
                    continue

                function_name = self.code_analyzer.get_function_name(node)

                # Try to match with sting function changes or create new one
                matching_key = None
                for key, func_change in function_changes.items():
                    if func_change.function_name == function_name:
                        matching_key = key
                        break

                if matching_key:
                    function_changes[matching_key].deleted_lines.append(deleted_line)
                else:
                    # Create new function change for deleted-only functions
                    function_code = self.code_analyzer.extract_node_source(previous_source_code, node)
                    if function_code:
                        func_key = f"{function_name}_{node.lineno}_deleted"
                        function_changes[func_key] = FunctionChange(
                            function_name=function_name,
                            function_code=function_code,
                            function_start_line=node.lineno,
                            function_end_line=node.end_lineno,
                            added_lines=[],
                            deleted_lines=[deleted_line],
                            sha=sha
                        )

        # Filter out functions with no meaningful changes
        return [func_change for func_change in function_changes.values() if func_change.has_changes]


class PRAnalyzerConfig:
    """Configuration class following SRP"""

    def __init__(self):
        self.github_repo = os.getenv("REPO")
        repo_parts = self.github_repo.split("/")
        self.organization_name = repo_parts[0]
        self.project_name = repo_parts[1]
        self.pr_number = os.getenv("PR_NUMBER")
        self.files_json = f"{self.organization_name}.{self.project_name}.{self.pr_number}_pr.json"
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.code_review_url = os.getenv("CODE_REVIEW_URL")


def convert_analysis_to_dict(analysis_result: Dict[str, List[FunctionChange]], project_name: str) -> Dict[str, Any]:
    """
    Convert PR analysis result to a structured dictionary format

    Args:
        analysis_result: The result from PRDiffAnalyzer.analyze_pr_changes()

    Returns:
        Dictionary with structured function changes information
    """
    result = {}

    for filename, function_changes in analysis_result.items():
        result[filename] = []

        for func_change in function_changes:
            # Process added lines to group consecutive lines
            added_ranges = _group_consecutive_lines(func_change.added_lines, is_added=True)

            # Process deleted lines to group consecutive lines
            deleted_ranges = _group_consecutive_lines(func_change.deleted_lines, is_added=False)

            function_data = {
                "file_path": filename,
                "project_name": project_name,
                "function_name": func_change.function_name,
                "function_location": {
                    "start_line": func_change.function_start_line,
                    "end_line": func_change.function_end_line
                },
                "full_function_code": func_change.function_code,
                "added_code": added_ranges,
                "deleted_code": deleted_ranges,
                "summary": {
                    "total_added_lines": len(func_change.added_lines),
                    "total_deleted_lines": len(func_change.deleted_lines),
                    "added_line_numbers": [line.line_number for line in func_change.added_lines],
                    "deleted_line_numbers": [line.original_line_number for line in func_change.deleted_lines]
                },
                "sha": func_change.sha
            }

            result[filename].append(function_data)

    return result


def _group_consecutive_lines(lines: List, is_added: bool = True) -> List[Dict[str, Any]]:
    """
    Group consecutive lines into ranges and return structured data

    Args:
        lines: List of AddedLine or DeletedLine objects
        is_added: True for added lines, False for deleted lines

    Returns:
        List of dictionaries with line ranges and code
    """
    if not lines:
        return []

    # Sort lines by line number
    if is_added:
        sorted_lines = sorted(lines, key=lambda x: x.line_number)
        get_line_num = lambda x: x.line_number
    else:
        sorted_lines = sorted(lines, key=lambda x: x.original_line_number)
        get_line_num = lambda x: x.original_line_number

    ranges = []
    current_range = {
        "start_line": get_line_num(sorted_lines[0]),
        "end_line": get_line_num(sorted_lines[0]),
        "code": sorted_lines[0].content,
        "line_count": 1
    }

    for i in range(1, len(sorted_lines)):
        current_line_num = get_line_num(sorted_lines[i])

        # Check if this line is consecutive to the previous one
        if current_line_num == current_range["end_line"] + 1:
            # Extend current range
            current_range["end_line"] = current_line_num
            current_range["code"] += "\n" + sorted_lines[i].content
            current_range["line_count"] += 1
        else:
            # Start new range
            ranges.append(current_range)
            current_range = {
                "start_line": current_line_num,
                "end_line": current_line_num,
                "code": sorted_lines[i].content,
                "line_count": 1
            }

    # Don't forget the last range
    ranges.append(current_range)

    return ranges


def print_formatted_result(result_dict: Dict[str, Any]) -> None:
    """
    Print the dictionary result in a formatted way

    Args:
        result_dict: The structured dictionary from convert_analysis_to_dict()
    """
    print("=== FORMATTED PR ANALYSIS RESULT ===\n")

    for filename, functions in result_dict.items():
        print(f"📁 File: {filename}")
        print("=" * 80)

        for i, func_data in enumerate(functions, 1):
            print(f"\n🔧 Function #{i}: {func_data['function_name']}")
            print(f"📍 Location: Lines {func_data['function_location']['start_line']}-{func_data['function_location']['end_line']}")

            # Summary
            summary = func_data['summary']
            print(f"📊 Summary: +{summary['total_added_lines']} lines, -{summary['total_deleted_lines']} lines")

            # Added code ranges
            if func_data['added_code']:
                print(f"\n✅ Added Code ({len(func_data['added_code'])} ranges):")
                for j, added_range in enumerate(func_data['added_code'], 1):
                    if added_range['start_line'] == added_range['end_line']:
                        print(f"  Range #{j}: Line {added_range['start_line']}")
                    else:
                        print(f"  Range #{j}: Lines {added_range['start_line']}-{added_range['end_line']}")
                    print(f"  Code:")
                    for line in added_range['code'].split('\n'):
                        print(f"    + {line}")

            # Deleted code ranges
            if func_data['deleted_code']:
                print(f"\n❌ Deleted Code ({len(func_data['deleted_code'])} ranges):")
                for j, deleted_range in enumerate(func_data['deleted_code'], 1):
                    if deleted_range['start_line'] == deleted_range['end_line']:
                        print(f"  Range #{j}: Line {deleted_range['start_line']}")
                    else:
                        print(f"  Range #{j}: Lines {deleted_range['start_line']}-{deleted_range['end_line']}")
                    print(f"  Code:")
                    for line in deleted_range['code'].split('\n'):
                        print(f"    - {line}")

            # Full function code (truncated for display)
            print(f"\n📋 Full Function Code:")
            function_lines = func_data['full_function_code'].split('\n')
            if len(function_lines) > 10:
                for line in function_lines[:5]:
                    print(f"    {line}")
                print(f"    ... ({len(function_lines) - 10} lines omitted) ...")
                for line in function_lines[-5:]:
                    print(f"    {line}")
            else:
                for line in function_lines:
                    print(f"    {line}")

            print("\n" + "-" * 60)

        print("\n")


async def request_code_review(config: PRAnalyzerConfig, payload: Dict[str, Any]) -> None:
    """
    Send code review request to an external service (e.g., LLM API)

    Args:
        payload: The structured data to send for code review
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(config.code_review_url, json=payload) as response:
            if response.status != 200:
                raise Exception(f"Failed to send code review request: {response.status}")
            return await response.json()


def construct_payload(payload, body, start_line, line, side):
    temp_payload = {
        "body": body,
        "commit_id": payload["sha"],
        "path": payload["file_path"],
        "side": side,
        "start_line": start_line,
        "line": line
    }

    if temp_payload["side"] == "LEFT":
        del temp_payload["start_line"]
    return temp_payload


def build_comments_payload(payloads, review_results):
    comments_payload = []
    for payload, review_result in zip(payloads, review_results):
        start_line = 0
        line = 0
        if len(payload["added_code"]) == 0:
            start_line = payload["deleted_code"][0]["start_line"]
            line = payload["deleted_code"][0]["end_line"]
        else:
            start_line = payload["added_code"][0]["start_line"]
            line = payload["added_code"][0]["end_line"]

        side = "LEFT" if len(payload["added_code"]) == 0 else "RIGHT"

        summary_review = review_result["summary"] if side == "RIGHT" else review_result["style_review"]
        duplication_review = ""
        reference_code = ""

        is_have_duplication_code = len(review_result["reference"]) > 0
        is_have_duplication_review = "no duplication" not in review_result["duplication_review"].lower()
        is_not_deleted_code = len(payload["added_code"]) > 0

        if is_have_duplication_code and is_have_duplication_review and is_not_deleted_code:
            duplication_review = review_result["duplication_review"]
            for similar_code in review_result["reference"]:
                similarity_score = float(similar_code["similarity"].strip('%'))
                if similarity_score > 80.0:
                    reference_code += f"""

similar code:
```
{similar_code["code"]}
```
similarity score: {similarity_score}%
file: {similar_code["file"]}
name: {similar_code["name"]}


                    """

            if duplication_review.strip() == "":
                body = summary_review
            else:
                body = f"""
{summary_review}

=================== DUPLICATION ANALYSIS ================
{duplication_review}

{reference_code}
                """
        else:
            body = summary_review

        temp_payload = construct_payload(
            payload=payload,
            body=body,
            start_line=start_line,
            line=line,
            side=side
        )
        comments_payload.append(temp_payload)

    return comments_payload


async def submit_review(config: PRAnalyzerConfig, comment_payload: Dict):
    url = f"https://api.github.com/repos/{config.github_repo}/pulls/{config.pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {config.github_token}",
        "Accept": "application/vnd.github+json"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=comment_payload, headers=headers) as response:
            return await response.json()


async def main():
    """Main function demonstrating usage"""
    config = PRAnalyzerConfig()

    # Dependency injection following DIP
    github_client = GitHubClient(config.github_token)
    file_storage = JSONFileStorage()
    patch_parser = UnidiffPatchParser()
    code_analyzer = ASTCodeAnalyzer()

    analyzer = PRDiffAnalyzer(
        github_client=github_client,
        file_storage=file_storage,
        patch_parser=patch_parser,
        code_analyzer=code_analyzer
    )

    # Download and save PR data
    analyzer.download_and_save_pr_data(
        config.github_repo,
        config.pr_number,
        config.files_json
    )

    # Analyze changes and get results
    analysis_result = analyzer.analyze_pr_changes(config.files_json, config.github_repo)
    structured_result = convert_analysis_to_dict(analysis_result, config.project_name)

    payloads = []
    for _, functions in structured_result.items():
        for function in functions:
            for added_code in function["added_code"]:
                if added_code["code"].strip() == "":
                    continue
                temp = dict(function)
                temp["added_code"] = [added_code]
                temp["deleted_code"] = []
                payloads.append(temp)

            for deleted_code in function["deleted_code"]:
                if deleted_code["code"].strip() == "":
                    continue
                temp = dict(function)
                temp["deleted_code"] = [deleted_code]
                temp["added_code"] = []
                payloads.append(temp)

    async with aiohttp.ClientSession():
        print("Starting ...")
        code_review_tasks = [request_code_review(config, payload) for payload in payloads]
        print("Reviewing code ...")
        review_results = await asyncio.gather(*code_review_tasks)
        comments_payload = build_comments_payload(payloads, review_results)
        submit_review_tasks = [submit_review(config, comment_payload) for comment_payload in comments_payload]
        print("Commenting the pr ...")
        await asyncio.gather(*submit_review_tasks)
        print("Done.")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
