# app/main.py
from github import Github
import requests, os

# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# REPO = "your-user/ai-pr-bot"

# g = Github(GITHUB_TOKEN)
# repo = g.get_repo(REPO)
# issues = repo.get_issues(state="open")

# for issue in issues:
#     prompt = f"Create a code patch based on the following issue:\n\n{issue.title}\n{issue.body}"
#     res = requests.post("http://llm-server:11434/generate", json={"prompt": prompt})
#     patch = res.json().get("response")

#     # Simulate writing patch to file (replace with real logic)
#     with open("/tmp/generated_patch.py", "w") as f:
#         f.write(patch)

#     # You could now commit this patch, push a branch, and create PR using GitHub API or GitPython
