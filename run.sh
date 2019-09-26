#!/bin/bash

# Capture command line arguments
TIFF_FILE=$1
OUTPUT=$2

echo ''
echo '----------------------------------------------------'
echo 'Calling job with arguments '"${TIFF_FILE}" "${OUTPUT}"
SCRIPT=app.py

python ${SCRIPT} "--index=NDVI" "--image-path=${TIFF_FILE}" "--verbose" "--output-directory=${OUTPUT}"
rc=$?
echo 'Done calling job - wrapper finished'
echo '----------------------------------------------------'
echo ''
exit ${rc}
