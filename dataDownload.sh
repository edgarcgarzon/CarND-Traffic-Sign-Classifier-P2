#Create folders
mkdir weights
mkdir traffic-signs-data

#Download the data
wget "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"

#Unzip the data 
unzip traffic-signs-data.zip -d traffic-signs-data

rm traffic-signs-data.zip
