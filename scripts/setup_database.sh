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

INPUT_ZIP="${PROJECT_ROOT}/nasa-cmaps.zip"
RAW_DATA_DIR="${PROJECT_ROOT}/data/raw"

printf "${B_CYAN}Project Root detected as: ${BOLD}%s${RESET}\n" "$PROJECT_ROOT"
printf "${B_CYAN}Starting dataset setup...${RESET}\n"

printf "${CYAN}Checking for zip file: ${BOLD}%s${RESET}... " "$INPUT_ZIP"
if [ ! -f "$INPUT_ZIP" ]; then
  printf "${B_RED}Not found!${RESET}\n"
  printf "${RED}Error: ${BOLD}%s${RESET}${RED} not found.${RESET}\n" "$INPUT_ZIP"
  printf "${YELLOW}Please run the download script first (e.g., ${BOLD}bash %s/download_dataset.sh${RESET}${YELLOW}).${RESET}\n" "$SCRIPT_DIR"
  exit 1
else
    printf "${GREEN}Found.${RESET}\n"
fi

printf "${CYAN}Ensuring raw data directory exists: ${BOLD}%s${RESET}... " "$RAW_DATA_DIR"
mkdir -p "$RAW_DATA_DIR"
if [ $? -ne 0 ]; then
    printf "${B_RED}Failed!${RESET}\n"
    printf "${RED}Error: Could not create directory ${BOLD}%s${RESET}${RED}.${RESET}\n" "$RAW_DATA_DIR"
    exit 1
else
    if [ -d "$RAW_DATA_DIR" ]; then
        printf "${GREEN}OK.${RESET}\n"
    else
        printf "${B_RED}Failed!${RESET}\n"
        printf "${RED}Error: Directory ${BOLD}%s${RESET}${RED} check failed after mkdir.${RESET}\n" "$RAW_DATA_DIR"
        exit 1
    fi
fi

printf "${CYAN}Unzipping ${BOLD}%s${RESET} into ${BOLD}%s${RESET} ...\n" "$INPUT_ZIP" "$RAW_DATA_DIR"
unzip -o "$INPUT_ZIP" -d "$RAW_DATA_DIR"

if [ $? -ne 0 ]; then
  printf "--------------------------------------------------\n"
  printf "${B_RED}Error:${RED} Unzipping failed.${RESET}\n"
  printf "${RED}The archive ${BOLD}%s${RESET}${RED} might be corrupted, or you might lack permissions.${RESET}\n" "$INPUT_ZIP"
  printf "--------------------------------------------------\n"
  exit 1
fi
printf "${GREEN}Unzipping completed successfully.${RESET}\n"

printf "${CYAN}Removing the zip file: ${BOLD}%s${RESET} ... " "$INPUT_ZIP"
rm "$INPUT_ZIP"
if [ $? -ne 0 ]; then
    printf "${B_YELLOW}Failed!${RESET}\n"
    printf "${YELLOW}Warning: Could not remove ${BOLD}%s${RESET}${YELLOW}. You may need to remove it manually.${RESET}\n" "$INPUT_ZIP"
else
    printf "${GREEN}Removed.${RESET}\n"
fi

printf "\n--------------------------------------------------\n"
printf "${B_GREEN}Dataset setup complete.${RESET}\n"
printf "${GREEN}Raw data should now be in: ${BOLD}%s${RESET}\n" "$RAW_DATA_DIR"
printf "${CYAN}Contents of ${BOLD}%s${RESET}:\n" "$RAW_DATA_DIR"
ls -lh "$RAW_DATA_DIR" 

exit 0
