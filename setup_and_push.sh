#!/bin/bash

echo "=================================================="
echo "Stanford RNA 3D Folding - Setup & Push to Kaggle"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}Step 1: Checking Kaggle API credentials...${NC}"
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}❌ Kaggle API credentials not found!${NC}"
    echo "Please download your kaggle.json from https://www.kaggle.com/settings"
    echo "and place it in ~/.kaggle/kaggle.json"
    exit 1
else
    echo -e "${GREEN}✓ Kaggle credentials found${NC}"
fi

echo -e "\n${YELLOW}Step 2: Testing Kaggle API...${NC}"
if ! kaggle competitions list -s rna &> /dev/null; then
    echo -e "${RED}❌ Kaggle API test failed. Please check your credentials.${NC}"
    echo "Visit: https://www.kaggle.com/settings/account"
    echo "Create a new API token and replace ~/.kaggle/kaggle.json"
    exit 1
else
    echo -e "${GREEN}✓ Kaggle API working${NC}"
fi

echo -e "\n${YELLOW}Step 3: Checking competition rules acceptance...${NC}"
echo "Attempting to access competition data..."
if ! kaggle competitions files stanford-rna-3d-folding-2 &> /dev/null; then
    echo -e "${RED}❌ You need to accept the competition rules!${NC}"
    echo "Visit: https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/rules"
    echo "Click 'I Understand and Accept'"
    echo ""
    echo "After accepting rules, run this script again."
    exit 1
else
    echo -e "${GREEN}✓ Competition rules accepted${NC}"
fi

echo -e "\n${YELLOW}Step 4: Checking kernel metadata...${NC}"
if [ ! -f kernel-metadata.json ]; then
    echo -e "${RED}❌ kernel-metadata.json not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Kernel metadata found${NC}"
fi

echo -e "\n${YELLOW}Step 5: Pushing kernel to Kaggle...${NC}"
kaggle kernels push -p .

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}=================================================="
    echo "✓ Kernel pushed successfully!"
    echo "==================================================${NC}"
    echo ""
    echo "Check status with:"
    echo "  kaggle kernels status aviralgarg05/stanford-rna-3d-folding-starter"
    echo ""
    echo "View output with:"
    echo "  kaggle kernels output aviralgarg05/stanford-rna-3d-folding-starter"
    echo ""
    echo "Monitor on Kaggle:"
    echo "  https://www.kaggle.com/code/aviralgarg05/stanford-rna-3d-folding-starter"
else
    echo -e "\n${RED}=================================================="
    echo "❌ Failed to push kernel"
    echo "==================================================${NC}"
    echo ""
    echo "Common issues:"
    echo "1. Competition rules not accepted"
    echo "2. API token expired or invalid"
    echo "3. Kernel already running (wait for it to finish)"
    echo ""
    echo "Try regenerating your API token:"
    echo "  https://www.kaggle.com/settings/account"
fi
