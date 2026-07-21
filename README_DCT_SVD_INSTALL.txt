AWSRE DCT-SVD INSTALLATION
==========================

1. Extract this package in the AWSRE repository root:

   unzip -o awsre_dct_svd_package.zip

2. Verify registration:

   python verify_dct_svd_install.py

3. Run the DCT-SVD test:

   python -m tests.test_dct_svd

4. Run the full smoke test:

   python -m tests.smoke_test

5. Commit:

   git add watermarking/dct_svd.py tests/test_dct_svd.py \
       verify_dct_svd_install.py README_DCT_SVD_INSTALL.txt
   git commit -m "Add DCT-SVD watermarking algorithm"
   git push
