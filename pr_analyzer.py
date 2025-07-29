from io import StringIO
import os
import json
import requests
import ast
import asttokens
from unidiff import PatchSet
from pathlib import Path
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
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


class GitHubAPIError(Exception):
    """Custom exception for GitHub API related errors"""
    pass


class CodeAnalysisError(Exception):
    """Custom exception for code analysis related errors"""
    pass


class IGitHubClient(ABC):
    """Interface for GitHub API operations"""

    @abstractmethod
    def get_pr_diff(self, repo: str, pr_number: int) -> str:
        pass

    @abstractmethod
    def get_pr_info(self, repo: str, pr_number: int) -> Dict:
        pass

    @abstractmethod
    def get_raw_file(self, repo: str, commit_sha: str, filepath: str) -> str:
        pass


class GitHubClient(IGitHubClient):
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
        """Download raw file content from GitHub"""
        url = f"https://raw.githubusercontent.com/{repo}/{commit_sha}/{filepath}"
        headers = {"Accept": "application/vnd.github.v3.raw"}

        response = self._make_request(url, headers)
        return response.text


class IFileStorage(ABC):
    """Interface for file storage operations"""

    @abstractmethod
    def save_files_data(self, data: List[FileData], filepath: str) -> None:
        pass

    @abstractmethod
    def load_files_data(self, filepath: str) -> List[FileData]:
        pass


class JSONFileStorage(IFileStorage):
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


class IPatchParser(ABC):
    """Interface for patch parsing operations"""

    @abstractmethod
    def parse_diff(self, diff_text: str) -> List[FileData]:
        pass

    @abstractmethod
    def extract_added_lines(self, patch: str) -> List[AddedLine]:
        pass


class UnidiffPatchParser(IPatchParser):
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

    def _get_removed_lines(self, patch: str) -> set:
        """Extract all removed lines to filter out moved code"""
        return {
            line[1:].strip() for line in patch.splitlines()
            if line.startswith("-") and not line.startswith("---")
        }

    def _extract_start_line(self, hunk_header: str) -> int:
        """Extract starting line number from hunk header"""
        parts = hunk_header.split(" ")
        for part in parts:
            if part.startswith("+"):
                return int(part[1:].split(",")[0]) - 1
        return 0

    def _is_added_line(self, line: str) -> bool:
        """Check if line is an added line (not header)"""
        return (line.startswith("+") and
                not line.startswith("+++") and
                not line.startswith("@@"))


class ICodeAnalyzer(ABC):
    """Interface for code analysis operations"""

    @abstractmethod
    def get_enclosing_node(self, source_code: str, target_line: int) -> Optional[ast.AST]:
        pass

    @abstractmethod
    def extract_node_source(self, source_code: str, node: ast.AST) -> Optional[str]:
        pass


class ASTCodeAnalyzer(ICodeAnalyzer):
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


class PRDiffAnalyzer:
    """Main class for analyzing PR diffs following SRP"""

    def __init__(
        self,
        github_client: IGitHubClient,
        file_storage: IFileStorage,
        patch_parser: IPatchParser,
        code_analyzer: ICodeAnalyzer
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

    def analyze_pr_changes(self, files_data_path: str, repo: str) -> None:
        """Analyze changes in PR and print normalized code"""
        files_data = self.file_storage.load_files_data(files_data_path)

        for file_data in files_data:
            added_lines = self.patch_parser.extract_added_lines(file_data.patch)

            if not added_lines:
                continue

            print(f"\n## File: {file_data.filename}\n")

            source_code = self.github_client.get_raw_file(repo, file_data.sha, file_data.filename)
            self._print_unique_functions(source_code, added_lines)

    def _print_unique_functions(self, source_code: str, added_lines: List[AddedLine]) -> None:
        """Print unique functions/classes that contain added lines"""
        printed_funcs = set()

        for added_line in added_lines:
            node = self.code_analyzer.get_enclosing_node(source_code, added_line.line_number)
            if not node:
                continue

            src = self.code_analyzer.extract_node_source(source_code, node)
            if src and src not in printed_funcs:
                printed_funcs.add(src)
                print(src)
                print("===")


class PRAnalyzerConfig:
    """Configuration class following SRP"""

    def __init__(self):
        self.github_repo = "mrrizal/django-exercise"
        self.pr_number = 1
        self.files_json = "files.json"
        self.github_token = os.getenv("GITHUB_KEY")


def main():
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

    try:
        # Download and save PR data
        analyzer.download_and_save_pr_data(
            config.github_repo,
            config.pr_number,
            config.files_json
        )

        # Analyze changes
        analyzer.analyze_pr_changes(config.files_json, config.github_repo)
        os.remove(config.files_json)
    except GitHubAPIError as e:
        print(f"GitHub API Error: {e}")
    except CodeAnalysisError as e:
        print(f"Code Analysis Error: {e}")
    except FileNotFoundError as e:
        print(f"File Error: {e}")


if __name__ == "__main__":
    main()
