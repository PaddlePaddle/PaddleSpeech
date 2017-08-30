cd noise 
python chime3_background.py
if [ $? -ne 0 ]; then
    echo "Prepare CHiME3 background noise failed. Terminated."
    exit 1
fi
cd -

cat noise/manifest.* > manifest.noise
echo "All done."
