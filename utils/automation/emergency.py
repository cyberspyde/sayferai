def activate_emergency():
    global emergency_active
    emergency_active = True
    print("Emergency activated")

def deactivate_emergency():
    global emergency_active
    emergency_active = False
    print("Emergency deactivated")