# Create directories
mkdir -p refcoco/images
mkdir -p refcoco/annotations

# Navigate to the images directory
cd refcoco/images

# Download COCO 2014 images (RefCOCO uses COCO 2014 images)
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip

# Unzip the downloaded files
unzip train2014.zip
unzip val2014.zip

# Remove zip files to conserve space
rm train2014.zip
rm val2014.zip

# Navigate to the annotations directory
cd ../annotations

# Download RefCOCO annotations
# Note: The annotations are hosted on external repositories; ensure you have access.
wget -c http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

# Unzip the annotations
unzip refcoco.zip

# Remove zip files to conserve space
rm refcoco.zip
