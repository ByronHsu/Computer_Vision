for i in {0..9};
do
   echo "Running Real $i"
   python3 -W ignore main.py --input-left data/Real/TL$i.bmp --input-right data/Real/TR$i.bmp --output ./output/r$i.pfm;
done

for i in {0..9};
do
    echo "Running Syn $i"
    python3 -W ignore main.py --input-left data/Synthetic/TL$i.png --input-right data/Synthetic/TR$i.png --output ./output/s$i.pfm;
done
