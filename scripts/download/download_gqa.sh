# Create directories
mkdir -p gqa/images
mkdir -p gqa/annotations

# Navigate to the images directory
cd gqa/images

# Download GQA images
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

# Unzip the downloaded files
unzip images.zip

# Remove zip files to conserve space
rm images.zip

# Navigate to the annotations directory
cd ../annotations

# Download GQA annotations
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip

# Unzip the annotations
unzip questions1.2.zip

# Remove zip files to conserve space
rm questions1.2.zip
