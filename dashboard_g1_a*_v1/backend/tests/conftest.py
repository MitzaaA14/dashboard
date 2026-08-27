import os


# Evită inițializarea DDS/SDK reală în sandbox înainte ca fake-urile testelor
# să poată fi instalate. `start_dashboard.sh` nu setează această variabilă.
os.environ.setdefault("G1_SKIP_SDK_INIT", "1")
