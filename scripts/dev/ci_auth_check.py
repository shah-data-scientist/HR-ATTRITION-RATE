import os
os.environ['API_KEY'] = 'test_secure_api_key_for_testing'
try:
    from api.auth import verify_password, get_password_hash, generate_api_key

    # Test password hashing
    password = "test_password_123"
    hashed = get_password_hash(password)
    if not verify_password(password, hashed):
        print("❌ Password verification failed")
        exit(1)
    if verify_password("wrong_password", hashed):
        print("❌ Should reject wrong password")
        exit(1)

    # Test API key generation
    api_key = generate_api_key()
    if len(api_key) != 64:
        print(f"❌ API key length wrong: {len(api_key)}")
        exit(1)

    print("✅ Authentication tests passed")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    exit(1)
