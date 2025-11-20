# Security Policy

## Verification

**Before running, verify SHA-256 hash:**

```cmd
certutil -hashfile FenotIT.exe SHA256
```

**Expected:**
```
c27498bd3eb7e260f2b5fab5c1d7a0042a3880c0612aaf0d1f42c4c8ae6eef19
```

If hash doesn't match, **DO NOT RUN**. Contact: c.tirado@cgiar.org

---

## Privacy

FenotIT does **NOT**:
- Send data over internet
- Require network connection
- Collect usage data or telemetry
- Access files outside selected folders
- Modify system files or registry

**100% local processing.**

---

## Permissions

**Requires:**
- Read access to selected image folders
- Write access to output folders

**Does NOT require:**
- Administrator rights
- Camera/microphone access
- Network access

---

## Report Issues

Email: c.tirado@cgiar.org  

**Do NOT** post publicly.

---

## Supported Versions

| Version | Support |
|---------|---------|
| 1.0.x   | ✅      |

---

Last Updated: January 20, 2025