from engine import GenosEngine

cmd = r"""&( ($psHome[21]+$psHome[30]+'x'))([System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('V3JpdGUtT3V0cHV0ICJwd24i')))"""

e = GenosEngine()

print("raw_cmd:", cmd)
print("is_obfuscated:", e.is_obfuscated(cmd))

deobf = e.deobfuscate_layer(cmd)
print("deobfuscated_cmd:", deobf)

result = e.scan(cmd)
print("scan_label:", result.get("label"))
print("scan_label_confidence:", result.get("label_confidence"))
print("scan_deobfuscated_cmd:", result.get("deobfuscated_cmd"))
print("scan_mitre_codes:", result.get("MITRE_codes"))