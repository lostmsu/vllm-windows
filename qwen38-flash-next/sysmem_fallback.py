"""Read or set the NVIDIA driver's "CUDA - Sysmem Fallback Policy" for python.exe.

Usage: python sysmem_fallback.py [--apply | --dry-apply | --undo] [--log FILE]
Without options it only reports. --apply and --undo need an elevated (admin)
process; --dry-apply makes the --apply changes in a session that is never saved.
"""
import ctypes as C
import sys
import traceback

# Hidden from NVAPI's public setting names; ID from nvidiaProfileInspector's
# CustomSettingNames.xml. Verify the effect with an over-VRAM allocation test.
SETTING_ID = 0x10ECECC9  # CUDA - Sysmem Fallback Policy
VALUE_NAMES = {0: "Default", 1: "Prefer no sysmem fallback", 2: "Prefer sysmem fallback"}
PREFER_NO_SYSMEM_FALLBACK = 1
APPS = ["python.exe"]
NEW_PROFILE = "Python CUDA (no sysmem fallback)"

NVAPI_OK = 0
NVAPI_PROFILE_NOT_FOUND = -163
NVAPI_EXECUTABLE_NOT_FOUND = -166
NVAPI_SETTING_NOT_FOUND = -160

US = C.c_wchar * 2048  # NvAPI_UnicodeString (NvU16[2048])


class Binary(C.Structure):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [("length", C.c_uint32), ("data", C.c_uint8 * 4096)]


class Value(C.Union):
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [("u32", C.c_uint32), ("binary", Binary), ("wsz", US),
                ("u64", C.c_uint64), ("f32", C.c_float)]


class Setting(C.Structure):  # NVDRS_SETTING_V1, #pragma pack(4)
    _layout_ = "ms"
    _pack_ = 4
    _fields_ = [("version", C.c_uint32), ("name", US), ("id", C.c_uint32),
                ("type", C.c_int), ("location", C.c_int),
                ("is_current_predefined", C.c_uint32),
                ("is_predefined_valid", C.c_uint32),
                ("predefined", Value), ("current", Value)]


class Profile(C.Structure):  # NVDRS_PROFILE_V1
    _fields_ = [("version", C.c_uint32), ("name", US), ("gpu_support", C.c_uint32),
                ("is_predefined", C.c_uint32), ("num_apps", C.c_uint32),
                ("num_settings", C.c_uint32)]


class Application(C.Structure):  # NVDRS_APPLICATION_V1
    _fields_ = [("version", C.c_uint32), ("is_predefined", C.c_uint32),
                ("app_name", US), ("friendly_name", US), ("launcher", US)]


assert C.sizeof(Setting) == 12320, C.sizeof(Setting)


def ustr(text):
    s = US()
    s.value = text
    return s


def version(struct):
    return C.sizeof(struct) | (1 << 16)


_nv = C.WinDLL("nvapi64.dll")
_qi = _nv.nvapi_QueryInterface
_qi.restype = C.c_void_p
_qi.argtypes = [C.c_uint32]


def _fn(fid, *argtypes):
    ptr = _qi(fid)
    if not ptr:
        raise OSError(f"NVAPI function {fid:#010x} not available")
    return C.CFUNCTYPE(C.c_int, *argtypes)(ptr)


def _internal_or_public(internal_id, internal_args, public_id, public_args, adapt):
    """The public DRS Get/SetSetting only accept settings named in
    NvApiDriverSettings.h; the internal variants (as used by
    nvidiaProfileInspector) also accept hidden ones."""
    if _qi(internal_id):
        return _fn(internal_id, *internal_args)
    public = _fn(public_id, *public_args)
    return lambda *args: public(*adapt(args))


H = C.c_void_p
P = C.POINTER
Initialize = _fn(0x0150E828)
GetErrorMessage = _fn(0x6C2D048C, C.c_int, C.c_char * 64)
CreateSession = _fn(0x0694D52E, P(H))
DestroySession = _fn(0xDAD9CFF8, H)
LoadSettings = _fn(0x375DBD6B, H)
SaveSettings = _fn(0xFCBC7E14, H)
GetBaseProfile = _fn(0xDA8466A0, H, P(H))
GetProfileInfo = _fn(0x61CD6FD6, H, H, P(Profile))
FindProfileByName = _fn(0x7E4A9A0B, H, P(US), P(H))
CreateProfile = _fn(0xCC176068, H, P(Profile), P(H))
FindApplicationByName = _fn(0xEEE566B2, H, P(US), P(H), P(Application))
CreateApplication = _fn(0x4347A9DE, H, H, P(Application))
DeleteProfile = _fn(0x17093206, H, H)
_GetSetting = _internal_or_public(
    0xEA99498D, (H, H, C.c_uint32, P(Setting), P(C.c_uint32)),
    0x73BF8338, (H, H, C.c_uint32, P(Setting)), lambda a: a[:4])
_SetSetting = _internal_or_public(
    0x8A2CF5F5, (H, H, P(Setting), C.c_uint32, C.c_uint32),
    0x577DD202, (H, H, P(Setting)), lambda a: a[:3])


def GetSetting(session, profile, setting_id, setting):
    return _GetSetting(session, profile, setting_id, setting, C.byref(C.c_uint32(0)))


def SetSetting(session, profile, setting):
    return _SetSetting(session, profile, setting, 0, 0)


def check(status, what):
    if status != NVAPI_OK:
        msg = (C.c_char * 64)()
        GetErrorMessage(status, msg)
        raise RuntimeError(f"{what} failed: {status} {msg.value.decode()}")


def read_setting(session, profile):
    s = Setting(version=version(Setting))
    status = GetSetting(session, profile, SETTING_ID, C.byref(s))
    if status == NVAPI_SETTING_NOT_FOUND:
        return "not set (inherits)"
    check(status, "GetSetting")
    where = {0: "this profile", 1: "global", 2: "base", 3: "driver default"}
    v = s.current.u32
    return f"{v} = {VALUE_NAMES.get(v, '?')} (from {where.get(s.location, s.location)})"


def profile_name(session, profile):
    p = Profile(version=version(Profile))
    check(GetProfileInfo(session, profile, C.byref(p)), "GetProfileInfo")
    return p.name + (" [predefined]" if p.is_predefined else "")


def find_app(session, app):
    profile = H()
    info = Application(version=version(Application))
    status = FindApplicationByName(session, C.byref(ustr(app)), C.byref(profile),
                                   C.byref(info))
    if status == NVAPI_EXECUTABLE_NOT_FOUND:
        return None
    check(status, f"FindApplicationByName({app})")
    return profile


def ensure_profile(session):
    profile = H()
    status = FindProfileByName(session, C.byref(ustr(NEW_PROFILE)), C.byref(profile))
    if status == NVAPI_PROFILE_NOT_FOUND:
        info = Profile(version=version(Profile), name=NEW_PROFILE,
                       gpu_support=0b111)  # geforce | quadro | nvs
        check(CreateProfile(session, C.byref(info), C.byref(profile)),
              "CreateProfile")
        print(f"created profile {NEW_PROFILE!r}")
    else:
        check(status, "FindProfileByName")
    return profile


def main():
    dry = "--dry-apply" in sys.argv
    apply = dry or "--apply" in sys.argv
    check(Initialize(), "NvAPI_Initialize")
    session = H()
    check(CreateSession(C.byref(session)), "CreateSession")
    try:
        check(LoadSettings(session), "LoadSettings")
        if "--undo" in sys.argv:
            profile = H()
            check(FindProfileByName(session, C.byref(ustr(NEW_PROFILE)),
                                    C.byref(profile)), "FindProfileByName")
            check(DeleteProfile(session, profile), "DeleteProfile")
            check(SaveSettings(session), "SaveSettings")
            print(f"deleted profile {NEW_PROFILE!r}")
            return
        base = H()
        check(GetBaseProfile(session, C.byref(base)), "GetBaseProfile")
        print(f"global (base profile): {read_setting(session, base)}")

        for app in APPS:
            profile = find_app(session, app)
            if profile is None:
                print(f"{app}: no profile")
                if not apply:
                    continue
                profile = ensure_profile(session)
                info = Application(version=version(Application), app_name=app,
                                   friendly_name=app)
                check(CreateApplication(session, profile, C.byref(info)),
                      f"CreateApplication({app})")
                print(f"{app}: added to {NEW_PROFILE!r}")
            print(f"{app}: profile {profile_name(session, profile)!r}, "
                  f"{read_setting(session, profile)}")
            if apply:
                s = Setting(version=version(Setting), id=SETTING_ID, type=0)
                s.current.u32 = PREFER_NO_SYSMEM_FALLBACK
                check(SetSetting(session, profile, C.byref(s)), "SetSetting")
        if dry:
            print("dry run: not saved")
        elif apply:
            check(SaveSettings(session), "SaveSettings")
            check(LoadSettings(session), "LoadSettings")
            for app in APPS:
                print(f"after save, {app}: "
                      f"{read_setting(session, find_app(session, app))}")
    finally:
        DestroySession(session)


if __name__ == "__main__":
    if "--log" in sys.argv:
        sys.stdout = sys.stderr = open(sys.argv[sys.argv.index("--log") + 1], "w")
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
