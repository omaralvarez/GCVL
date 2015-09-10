#!/bin/bash

REPO_PATH=git@github.com:omaralvarez/GCVL.git
HTML_PATH=doc/html
CLONE_PATH=gh-pages
COMMIT_USER="travis-ci"
COMMIT_EMAIL="travis@travis-ci.org"
CHANGESET=$(git rev-parse --verify HEAD)

echo -e "Publishing doxygen...\n"

mkdir -p ${HTML_PATH}
git clone -b gh-pages "${REPO_PATH}" --single-branch ${CLONE_PATH}

doxygen Doxyfile

cd ${CLONE_PATH}
cp -Rf ${HTML_PATH} ./doxygen
git add -f .
git config user.name "${COMMIT_USER}"
git config user.email "${COMMIT_EMAIL}"
git commit -m "Automated documentation build for changeset ${CHANGESET}."
git push origin gh-pages
cd -
