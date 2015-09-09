#!/bin/bash

REPO_PATH=git@github.com:omaralvarez/GCVL.git
HTML_PATH=doc/html
COMMIT_USER="travis-ci"
COMMIT_EMAIL="travis@travis-ci.org"
CHANGESET=$(git rev-parse --verify HEAD)

echo -e "Publishing doxygen...\n"

rm -rf ${HTML_PATH}
mkdir -p ${HTML_PATH}
git clone -b gh-pages "${REPO_PATH}" --single-branch ${HTML_PATH}

cd ${HTML_PATH}
git rm -rf .
cd -

doxygen Doxyfile

cd ${HTML_PATH}
git add .
git config user.name "${COMMIT_USER}"
git config user.email "${COMMIT_EMAIL}"
git commit -m "Automated documentation build for changeset ${CHANGESET}."
git push origin gh-pages
cd -