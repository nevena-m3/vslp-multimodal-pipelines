# Security and Sensitive-Data Handling

## Scope

VSLP processes audio, video, metadata, clinical labels, and derived measurements that may be identifiable or sensitive. Operators are responsible for institutional approvals, access control, retention, and permitted use.

## Local-First Processing

The applications are designed to process files locally. Do not assume that a dependency, pretrained model installer, package manager, cloud-synced folder, or operating-system service satisfies study data-governance requirements.

## Never Commit

- participant audio or video;
- metadata containing participant identifiers;
- generated project outputs from real studies;
- access tokens, credentials, private URLs, or secrets;
- proprietary model files without redistribution permission;
- absolute local paths that reveal sensitive storage structure.

Use ignored local directories or approved external storage.

## Reporting a Vulnerability

Do not open a public issue containing sensitive details. Contact the repository maintainers and the Speech Production Lab through approved institutional channels. Include the affected version/commit, reproduction steps using synthetic data, impact, and proposed mitigation if known.

## Operational Recommendations

- Use encrypted, access-controlled storage.
- Keep raw data read-only where practical.
- Separate source media from generated outputs.
- Review export packages before sharing.
- Remove unnecessary source paths and direct identifiers from external handoffs.
- Pin and review dependencies for controlled deployments.
- Validate downloaded model files and licenses.
- Preserve manifests and logs without exposing participant data.

## Clinical Safety

Security does not imply clinical validity. VSLP is research software and is not approved for diagnosis or treatment decisions.
