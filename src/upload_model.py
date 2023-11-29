import argparse
import os

from dagshub.upload import Repo

def main():
    parser = argparse.ArgumentParser('Upload a new model to the repo and track it using DVC')
    parser.add_argument('--model', help='Model to upload')
    parser.add_argument('--user', help='DagsHub username')
    parser.add_argument('--token', help='Access token to DagsHub')
    parser.add_argument('--commit', required=False, help='Optional commit message')

    args = parser.parse_args()

    remote = os.popen('dvc remote list').readline()
    remote = remote.split()[-1]
    repo_owner, repo_name = os.path.split(remote)
    repo_name = os.path.splitext(repo_name)[0]
    _, repo_owner = os.path.split(repo_owner)

    repo = Repo(repo_owner, repo_name, args.user, args.token)

    commit_msg = args.commit or f"Add {os.path.split(args.model)[-1]} to the models folder."

    repo.upload(args.model, commit_msg)

if __name__ == '__main__':
    main()
