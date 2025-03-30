# Create directories
mkdir -p vqa/images
mkdir -p vqa/annotations

# Navigate to the images directory
cd vqa/images

# Download COCO images (VQA uses COCO images)
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2015.zip

# Unzip the downloaded files
unzip train2014.zip
unzip val2014.zip
unzip test2015.zip

# Remove zip files to conserve space
rm train2014.zip
rm val2014.zip
rm test2015.zip

# Navigate to the annotations directory
cd ../annotations

# Download VQA annotations
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

# Unzip the annotations
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
unzip v2_Questions_Test_mscoco.zip

# Remove zip files to conserve space
rm v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Val_mscoco.zip
rm v2_Questions_Train_mscoco.zip
rm v2_Questions_Val_mscoco.zip
rm v2_Questions_Test_mscoco.zip
