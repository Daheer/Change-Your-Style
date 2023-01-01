read -p "Enter name for your environment" ENV_NAME
python venv ENV_NAME
cd ENV_NAME
source bin/activate
python -m pip --upgrade pip
python -m pip install -r requirements.txt
#read -p "Optional: download models to disk? (y/n)" DLOAD
#if [ "$DLOAD" = "y" || "$DLOAD" = "Y" || "$DLOAD" = "" ] then
uvicorn main:ap