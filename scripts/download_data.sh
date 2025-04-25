#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( dirname "$SCRIPT_DIR" )

RESET='\033[0m'       
BOLD='\033[1m'
RED='\033[0;31m'      # Red
GREEN='\033[0;32m'    # Green
YELLOW='\033[0;33m'   # Yellow
BLUE='\033[0;34m'     # Blue
CYAN='\033[0;36m'     # Cyan
B_RED='\033[1;31m'     # Bold Red
B_GREEN='\033[1;32m'   # Bold Green
B_YELLOW='\033[1;33m'  # Bold Yellow
B_CYAN='\033[1;36m'    # Bold Cyan

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/behrad3d/nasa-cmaps"
OUTPUT_ZIP="${PROJECT_ROOT}/nasa-cmaps.zip"

printf "${B_CYAN}Project Root detected as: ${BOLD}%s${RESET}\n" "$PROJECT_ROOT"
printf "${B_CYAN}Starting dataset download...${RESET}\n"
printf "URL: ${BOLD}%s${RESET}\n" "$DATASET_URL"
printf "Outputting to: ${BOLD}%s${RESET}\n" "$OUTPUT_ZIP"

if [ -f "$OUTPUT_ZIP" ]; then
  printf "${B_YELLOW}Warning:${YELLOW} %s already exists. Skipping download.${RESET}\n" "$OUTPUT_ZIP"
  printf "${YELLOW}If you need to re-download, please delete the existing file first.${RESET}\n"
  exit 0 
fi

printf "${CYAN}Executing download command...${RESET}\n"
curl -L -o "$OUTPUT_ZIP" "$DATASET_URL"

if [ $? -ne 0 ]; then
  printf "\n--------------------------------------------------\n"
  printf "${B_RED}Error:${RED} Download failed.${RESET}\n"
  printf "${RED}Please check the URL and your internet connection.${RESET}\n"
  printf "${RED}If the URL requires login, this curl command might not work.${RESET}\n"
  printf "${YELLOW}Consider using the official Kaggle CLI: ${BOLD}kaggle datasets download -d behrad3d/nasa-cmaps -p \"%s\"${RESET}\n" "$PROJECT_ROOT"
  printf "--------------------------------------------------\n"
  printf "${CYAN}Cleaning up potentially incomplete file: %s${RESET}\n" "$OUTPUT_ZIP"
  rm -f "$OUTPUT_ZIP"
  exit 1
else
  printf "\n--------------------------------------------------\n"
  printf "${B_GREEN}Dataset downloaded successfully as ${BOLD}%s${RESET}\n" "$OUTPUT_ZIP"
  printf "--------------------------------------------------\n"
fi

exit 0

