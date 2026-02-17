@echo off
REM Ripristina il repository allo stato dell'ultimo commit e rimuove file non tracciati.
git reset --hard HEAD
git clean -fd

echo Repository ripristinato.
