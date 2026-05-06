from __future__ import annotations

PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


DAMAGE_TOOL_MAP: dict[str, list[dict]] = {
    "front_damage": [
        {"category": "Cutters", "tool": "Hydraulic Cutter", "priority": "CRITICAL", "reason": "Front impact may deform the A-pillar, hood, and firewall; cutters create access."},
        {"category": "Combi Tools", "tool": "Combi Cutter-Spreader", "priority": "CRITICAL", "reason": "Useful when both cutting and spreading are needed in the front section."},
        {"category": "Rescue Rams", "tool": "Hydraulic Rescue Ram", "priority": "CRITICAL", "reason": "Can push back dashboard or firewall intrusion toward the engine bay."},
        {"category": "Spreaders", "tool": "Hydraulic Spreader (Jaws of Life)", "priority": "HIGH", "reason": "Helps open jammed front doors and structural gaps."},
        {"category": "Specialised Tools", "tool": "Seatbelt Cutter / Hook Knife", "priority": "HIGH", "reason": "Seatbelts may be locked or trapped after impact."},
        {"category": "Specialised Tools", "tool": "Cervical Collar + Spine Board (KED)", "priority": "CRITICAL", "reason": "Frontal impacts can cause high spinal injury risk."},
    ],
    "side_damage": [
        {"category": "Spreaders", "tool": "Hydraulic Spreader (Jaws of Life)", "priority": "CRITICAL", "reason": "Side impacts commonly jam doors; spreaders open the access path."},
        {"category": "Cutters", "tool": "Hydraulic Cutter", "priority": "CRITICAL", "reason": "Cuts hinges or pillars when door removal is needed."},
        {"category": "Combi Tools", "tool": "Combi Cutter-Spreader", "priority": "CRITICAL", "reason": "Combines cutting and spreading in a single rescue device."},
        {"category": "Specialised Tools", "tool": "Glass Management Tool (Auto Punch)", "priority": "HIGH", "reason": "Side windows often need controlled removal before door work."},
        {"category": "Specialised Tools", "tool": "Seatbelt Cutter / Hook Knife", "priority": "HIGH", "reason": "Used if occupant belts are locked or inaccessible."},
        {"category": "Specialised Tools", "tool": "Cervical Collar + Spine Board (KED)", "priority": "CRITICAL", "reason": "Lateral impacts create strong spinal injury risk."},
        {"category": "Lifting Equipment", "tool": "Vehicle Stabilisation Kit", "priority": "CRITICAL", "reason": "Prevents movement during extrication."},
    ],
    "rear_damage": [
        {"category": "Cutters", "tool": "Hydraulic Cutter", "priority": "HIGH", "reason": "Helps clear crushed rear metal and create access."},
        {"category": "Spreaders", "tool": "Mini Spreader (Compact)", "priority": "HIGH", "reason": "Useful for tight rear gaps and confined spaces."},
        {"category": "Specialised Tools", "tool": "Seatbelt Cutter / Hook Knife", "priority": "HIGH", "reason": "Useful when rear occupants are restrained or trapped."},
        {"category": "Specialised Tools", "tool": "Cervical Collar + Spine Board (KED)", "priority": "HIGH", "reason": "Rear impacts can still cause spinal injuries."},
        {"category": "Lifting Equipment", "tool": "Vehicle Stabilisation Kit", "priority": "CRITICAL", "reason": "Stabilization is required before rear access operations."},
    ],
    "pillar_damage": [
        {"category": "Cutters", "tool": "Hydraulic Cutter", "priority": "CRITICAL", "reason": "Pillar damage may require structural cutting for access."},
        {"category": "Spreaders", "tool": "Hydraulic Spreader (Jaws of Life)", "priority": "CRITICAL", "reason": "Helps fold back cut roof sections."},
        {"category": "Lifting Equipment", "tool": "Vehicle Stabilisation Kit", "priority": "CRITICAL", "reason": "Structural damage makes stabilization mandatory."},
        {"category": "Rescue Rams", "tool": "Hydraulic Rescue Ram", "priority": "CRITICAL", "reason": "Can push back dashboard or firewall intrusion toward the engine bay."},
        {"category": "Specialised Tools", "tool": "Cervical Collar + Spine Board (KED)", "priority": "CRITICAL", "reason": "Pillar collapse indicates high spinal risk."},
        {"category": "Lifting Equipment", "tool": "Vehicle Stabilisation Kit", "priority": "CRITICAL", "reason": "Must stabilize before roof operations."},
    ],
    "roof_damage": [
        {"category": "Cutters", "tool": "Hydraulic Cutter", "priority": "CRITICAL", "reason": "Roof removal often requires cutting multiple pillars or rails."},
        {"category": "Combi Tools", "tool": "Combi Cutter-Spreader", "priority": "CRITICAL", "reason": "Useful for roof flap or full roof removal."},
        {"category": "Spreaders", "tool": "Hydraulic Spreader (Jaws of Life)", "priority": "CRITICAL", "reason": "Helps fold back cut roof sections."},
        {"category": "Rescue Rams", "tool": "Hydraulic Rescue Ram", "priority": "CRITICAL", "reason": "Can create vertical space in severe roof crush cases."},
        {"category": "Specialised Tools", "tool": "Seatbelt Cutter / Hook Knife", "priority": "HIGH", "reason": "Belts may be locked after roof crush."},
        {"category": "Specialised Tools", "tool": "Cervical Collar + Spine Board (KED)", "priority": "CRITICAL", "reason": "Roof crush has high spinal injury risk."},
        {"category": "Lifting Equipment", "tool": "Vehicle Stabilisation Kit", "priority": "CRITICAL", "reason": "Must stabilize before roof operations."},
    ],
    "glass_break": [
        {"category": "Specialised Tools", "tool": "Glass Management Tool (Auto Punch)", "priority": "HIGH", "reason": "Safely clears broken glass before rescue work."},
        {"category": "Specialised Tools", "tool": "Seatbelt Cutter / Hook Knife", "priority": "HIGH", "reason": "Useful if belts are locked or occupant is suspended by belt."},
        {"category": "Lifting Equipment", "tool": "Vehicle Stabilisation Kit", "priority": "CRITICAL", "reason": "Vehicle should be stabilized before glass removal."},
    ],
    "dent": [
        {"category": "Specialised Tools", "tool": "Dent Puller / Slide Hammer", "priority": "LOW", "reason": "Minor dents may block access without requiring heavy hydraulic tools."},
        {"category": "Lifting Equipment", "tool": "Vehicle Stabilisation Kit", "priority": "LOW", "reason": "Basic stabilization is still good practice."},
    ],
    
}



CATEGORY_ORDER = [
    "Cutters",
    "Spreaders",
    "Combi Tools",
    "Rescue Rams",
    "Specialised Tools",
    "Power Units",
    "Hoses",
    "Lifting Equipment",
]

def recommend_tools(detection_result: dict) -> dict:
    summary = detection_result.get("summary", {})
    damage_types = summary.get("damage_types", [])
    severity_levels = summary.get("severity_levels", [])
    battery_hazard = detection_result.get("battery_hazard", False)

    active_labels = list(dict.fromkeys(damage_types))
    if battery_hazard:
        active_labels.append("battery_hazard")

    if not active_labels:
        return {
            "recommended_tools": [],
            "detected_damages": [],
            "severity_levels": [],
            "summary_message": "No major damage detected. Perform a manual scene check.",
            "annotated_image": detection_result.get("annotated_image"),
        }

    seen = {}
    for label in active_labels:
        for entry in DAMAGE_TOOL_MAP.get(label, []):
            tool_name = entry["tool"]
            if tool_name not in seen:
                seen[tool_name] = dict(entry)
            else:
                current = seen[tool_name]
                if PRIORITY_ORDER.get(entry["priority"], 99) < PRIORITY_ORDER.get(current["priority"], 99):
                    seen[tool_name] = dict(entry)

    tools = sorted(
        seen.values(),
        key=lambda x: (
            PRIORITY_ORDER.get(x["priority"], 99),
            CATEGORY_ORDER.index(x["category"]) if x["category"] in CATEGORY_ORDER else 99,
            x["tool"],
        ),
    )

    summary_parts = []
    if battery_hazard:
        summary_parts.append("Battery hazard detected: isolate HV system before any cutting.")
    summary_parts.append(f"Detected damages: {', '.join(active_labels)}.")
    summary_parts.append(
        f"Recommended {len(tools)} tools across {len(set(t['category'] for t in tools))} categories."
    )

    return {
        "recommended_tools": tools,
        "detected_damages": damage_types,
        "severity_levels": severity_levels,
        "battery_hazard": battery_hazard,
        "critical_found": detection_result.get("critical_found", False),
        "annotated_image": detection_result.get("annotated_image"),
        "summary_message": " ".join(summary_parts),
    }