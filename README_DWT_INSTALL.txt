AWSRE DWT PACKAGE
=================

1. Extract this ZIP into the root of AWSRE-Framework.
2. Install dependency:

   pip install PyWavelets

3. Run installer:

   python install_dwt.py

4. Verify:

   python verify_dwt_install.py
   python -m tests.test_dwt
   python -m tests.smoke_test

5. Commit and push:

   git add watermarking/dwt.py watermarking/__init__.py tests/test_dwt.py install_dwt.py verify_dwt_install.py
   git commit -m "Add DWT watermarking algorithm"
   git push origin main

Important:
- registry.py already contains watermarking.dwt, so the installer will not duplicate it.
- tests/test_dwt.py uses the actual registry API: list_registered_methods().
