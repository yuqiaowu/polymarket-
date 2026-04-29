from __future__ import annotations

from market_system.report import build_report, write_reports


def main() -> None:
    report = build_report()
    json_path, md_path = write_reports(report)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    candidate = report.get("candidate_layer", {}).get("primary_candidate", {})
    permission = report.get("trade_permission", {})
    print(
        "Primary candidate: "
        f"{candidate.get('action')} {candidate.get('target_symbol') or 'NONE'} "
        f"({candidate.get('strategy')}, rule={candidate.get('rule_status')})"
    )
    print(
        "Trade permission: "
        f"{permission.get('open_permission')} / {permission.get('direction_permission')} / {permission.get('position_size')}"
    )


if __name__ == "__main__":
    main()
