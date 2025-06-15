#!/bin/sh
cd /app

npm install

echo "Starting dev server..."
npm run dev -- --host
