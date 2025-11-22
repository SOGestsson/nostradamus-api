#!/bin/bash

# Test script for Inventory Simulation API
# Usage: ./test_api.sh

BASE_URL="http://localhost:8000"
DATA_FILE="/Users/siggi/Dropbox/2025/python/Nostradamus_api/all_sim_input_data.v2.json"

echo "Testing Inventory Simulation API"
echo "================================="
echo ""

# Test 1: Health check
echo "1. Testing health check endpoint..."
curl -s "${BASE_URL}/health" | jq '.'
echo ""

# Test 2: Root endpoint
echo "2. Testing root endpoint..."
curl -s "${BASE_URL}/" | jq '.'
echo ""

# Test 3: Full simulation (with histogram)
echo "3. Testing /api/v1/simulation/simulate endpoint..."
echo "   (This may take 30-60 seconds...)"
curl -s -X POST "${BASE_URL}/api/v1/simulation/simulate" \
  -H "Content-Type: application/json" \
  -d @"${DATA_FILE}" | jq '.histogram_info'
echo ""

# Test 4: Histogram only
echo "4. Testing /api/v1/simulation/histo_buy endpoint..."
echo "   (This may take 30-60 seconds...)"
curl -s -X POST "${BASE_URL}/api/v1/simulation/histo_buy" \
  -H "Content-Type: application/json" \
  -d @"${DATA_FILE}" | jq '.histogram_info'
echo ""

echo "Tests completed!"
