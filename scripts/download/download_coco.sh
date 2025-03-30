# Create directories
mkdir -p coco/images
cd coco/images

# Download image datasets
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/test2017.zip

# Unzip the downloaded files
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

# Remove zip files to conserve space
rm train2017.zip
rm val2017.zip
rm test2017.zip

# Navigate back to the coco directory
cd ..

# Download annotation files
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip annotation files
unzip annotations_trainval2017.zip

# Remove zip files to conserve space
rm annotations_trainval2017.zip
