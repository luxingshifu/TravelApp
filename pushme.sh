#!/bin/sh

git add $1
git commit -m "Debuigging $1"
git push heroku master
