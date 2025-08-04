import re
from typing import List, Dict, Tuple

def parse_diff(diff_content: str) -> Dict[str, List[Dict]]:
    """
    Parse git diff and extract line information for GitHub PR API comments.

    Returns:
        Dictionary with file paths as keys and list of changes as values.
        Each change contains: type, start_line, end_line, content
    """
    files = {}
    current_file = None
    current_old_line = 0
    current_new_line = 0

    lines = diff_content.strip().split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Parse file header
        if line.startswith('diff --git'):
            # Extract file path from the diff header
            match = re.search(r'diff --git a/(.*?) b/(.*?)$', line)
            if match:
                current_file = match.group(2)  # Use the 'b/' path (after changes)
                files[current_file] = []

        # Parse hunk header (@@)
        elif line.startswith('@@'):
            # Extract line numbers from hunk header
            # Format: @@ -old_start,old_count +new_start,new_count @@
            match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                current_old_line = int(match.group(1))
                current_new_line = int(match.group(3))

        # Parse content lines
        elif current_file and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
            if line.startswith('+') and not line.startswith('+++'):
                # Added line
                files[current_file].append({
                    'type': 'addition',
                    'line_number': current_new_line,
                    'content': line[1:]  # Remove the '+' prefix
                })
                current_new_line += 1

            elif line.startswith('-') and not line.startswith('---'):
                # Deleted line
                files[current_file].append({
                    'type': 'deletion',
                    'line_number': current_old_line,
                    'content': line[1:]  # Remove the '-' prefix
                })
                current_old_line += 1

            elif line.startswith(' '):
                # Context line (unchanged)
                current_old_line += 1
                current_new_line += 1

        i += 1

    return files

def group_consecutive_changes(changes: List[Dict]) -> List[Dict]:
    """
    Group consecutive additions/deletions for more efficient PR comments.
    """
    if not changes:
        return []

    grouped = []
    current_group = {
        'type': changes[0]['type'],
        'start_line': changes[0]['line_number'],
        'end_line': changes[0]['line_number'],
        'content': [changes[0]['content']]
    }

    for change in changes[1:]:
        # If same type and consecutive line
        if (change['type'] == current_group['type'] and
            change['line_number'] == current_group['end_line'] + 1):
            current_group['end_line'] = change['line_number']
            current_group['content'].append(change['content'])
        else:
            # Start new group
            grouped.append(current_group)
            current_group = {
                'type': change['type'],
                'start_line': change['line_number'],
                'end_line': change['line_number'],
                'content': [change['content']]
            }

    # Add the last group
    grouped.append(current_group)
    return grouped

def format_for_github_api(parsed_diff: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Format the parsed diff for GitHub PR Review API.
    Returns list of comment objects ready for GitHub API.
    """
    comments = []

    for file_path, changes in parsed_diff.items():
        # Group consecutive changes
        additions = [c for c in changes if c['type'] == 'addition']
        deletions = [c for c in changes if c['type'] == 'deletion']

        # Group consecutive additions
        if additions:
            grouped_additions = group_consecutive_changes(additions)
            for group in grouped_additions:
                comment = {
                    'path': file_path,
                    'line': group['end_line'],  # For single line comments
                    'start_line': group['start_line'] if group['start_line'] != group['end_line'] else None,
                    'side': 'RIGHT',  # RIGHT for additions, LEFT for deletions
                    'body': f"Added lines {group['start_line']}-{group['end_line']}" if group['start_line'] != group['end_line'] else f"Added line {group['end_line']}"
                }
                # Remove start_line if it's the same as line (single line comment)
                if comment['start_line'] == comment['line']:
                    comment['start_line'] = None
                comments.append(comment)

        # Group consecutive deletions
        if deletions:
            grouped_deletions = group_consecutive_changes(deletions)
            for group in grouped_deletions:
                comment = {
                    'path': file_path,
                    'line': group['end_line'],
                    'start_line': group['start_line'] if group['start_line'] != group['end_line'] else None,
                    'side': 'LEFT',  # LEFT for deletions
                    'body': f"Deleted lines {group['start_line']}-{group['end_line']}" if group['start_line'] != group['end_line'] else f"Deleted line {group['end_line']}"
                }
                if comment['start_line'] == comment['line']:
                    comment['start_line'] = None
                comments.append(comment)

    return comments

# Example usage with your diff
if __name__ == "__main__":
    diff_content = """diff --git a/.github/workflows/pr-check.yml b/.github/workflows/pr-check.yml
index aacd870..c8fce4f 100644
--- a/.github/workflows/pr-check.yml
+++ b/.github/workflows/pr-check.yml
@@ -31,3 +31,12 @@ jobs:
       - name: Run tests
         run: |
           python manage.py test
+
+  code_review:
+    name: Code Review
+    uses: mrrizal/python-pr-analyzer/.github/workflows/pr_analyzer.yml@main
+    with:
+      repository: ${{ github.repository }}
+      pr_number: ${{ github.event.pull_request.number }}
+    secrets:
+      token: ${{ secrets.GITHUB_TOKEN }}
diff --git a/product_service/views.py b/product_service/views.py
index 86b8d4a..808530c 100644
--- a/product_service/views.py
+++ b/product_service/views.py
@@ -27,19 +27,51 @@ def create(self, request, *args, **kwargs):

         return Response({"status": STATUS_SUCCESS, "message": message}, status=201)

-    def list(self, request, *args, **kwargs):
-        queryset = self.filter_queryset(self.get_queryset())
-        self.serializer_class = ProductLimitVariantsSerializer
+    def filter_queryset_updated(self, request, queryset):
+        \"\"\"
+        Override this method to apply custom filtering logic.
+        \"\"\"
+        empty_result = {
+            "next": None,
+            "previous": None,
+            "results": []
+        }

         datetime_format = "%d-%m-%YT%H:%M:%S"
         created_at_gte = request.GET.get('created_at_gte', None)
         created_at_lte = request.GET.get('created_at_lte', None)

+        if created_at_gte:
+            try:
+                created_at_gte = to_indonesia_timezone(
+                    f'{created_at_gte}T00:00:00', datetime_format)
+                queryset = queryset.filter(created_at__gte=created_at_gte)
+            except ValueError:
+                return Response(empty_result)
+
+        if created_at_lte:
+            try:
+                created_at_lte = to_indonesia_timezone(
+                    f'{created_at_lte}T23:59:59', datetime_format)
+                queryset = queryset.filter(created_at__lte=created_at_lte)
+            except ValueError:
+                return Response(empty_result)
+        return queryset
+
+    def list(self, request, *args, **kwargs):
+        queryset = self.filter_queryset(self.get_queryset())
+        self.serializer_class = ProductLimitVariantsSerializer
+
         empty_result = {
             "next": None,
             "previous": None,
             "results": []
         }
+
+        datetime_format = "%d-%m-%YT%H:%M:%S"
+        created_at_gte = request.GET.get('created_at_gte', None)
+        created_at_lte = request.GET.get('created_at_lte', None)
+
         if created_at_gte:
             try:
                 created_at_gte = to_indonesia_timezone("""

    # Parse the diff
    parsed = parse_diff(diff_content)

    # Print parsed results
    print("=== PARSED DIFF ===")
    for file_path, changes in parsed.items():
        print(f"\nFile: {file_path}")
        for change in changes:
            print(f"  {change['type']}: Line {change['line_number']} - {change['content'][:50]}...")

    # Format for GitHub API
    github_comments = format_for_github_api(parsed)

    print("\n=== GITHUB API FORMAT ===")
    for comment in github_comments:
        print(f"File: {comment['path']}")
        print(f"  Line: {comment['line']}")
        if comment['start_line']:
            print(f"  Start Line: {comment['start_line']}")
        print(f"  Side: {comment['side']}")
        print(f"  Body: {comment['body']}")
        print()
