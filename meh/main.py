from lxml import etree


def calculate_reduced_inertia(original_inertia, reduction_ratio):
    return original_inertia / (reduction_ratio ** 2)

def calculate_cylinder_inertia(mass, radius, height):
    Ixx = (1 / 12) * mass * (3 * radius ** 2 + height ** 2)
    Iyy = Ixx
    Izz = (1 / 2) * mass * radius ** 2
    return Ixx, Iyy, Izz


input_file = "ur3.xml"
output_file = "ur3_modified.xml"
tree = etree.parse(input_file)
root = tree.getroot()

harmonic_drives_catalog = [
    {"model": "HD1", "reduction_ratio": 100, "max_torque": 100, "inertia": 0.024},
    {"model": "HD2", "reduction_ratio": 160, "max_torque": 120, "inertia": 0.05},
    {"model": "HD3", "reduction_ratio": 200, "max_torque": 150, "inertia": 0.089},
]

reflected_inertia = 0.01

joints = root.findall(".//joint")

for joint in joints:
    joint.set("armature", str(reflected_inertia))

bodies = root.findall(".//body")
for body in bodies:
    inertial = body.find("inertial")
    if inertial is not None:
        mass = float(inertial.get("mass", "1.0"))
        new_mass = mass * 1.1
        inertial.set("mass", str(new_mass))

        diaginertia = inertial.get("diaginertia", "0.01 0.01 0.01").split()
        new_diaginertia = [str(float(val) * 1.2) for val in diaginertia]
        inertial.set("diaginertia", " ".join(new_diaginertia))

actuator = root.find("actuator")
if actuator is None:
    actuator = etree.Element("actuator")
    root.append(actuator)

ctrlrange = [-100, 100]
for joint in joints:
    motor = etree.Element("motor")
    motor.set("name", f"{joint.get('name')}_motor")
    motor.set("joint", joint.get("name"))
    motor.set("ctrlrange", f"{ctrlrange[0]} {ctrlrange[1]}")
    actuator.append(motor)

tree.write(output_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

variant_files = []
for drive in harmonic_drives_catalog:
    for joint in joints:
        original_inertia = float(joint.get("armature", "0.01"))
        reduced_inertia = calculate_reduced_inertia(original_inertia, drive["reduction_ratio"])
        joint.set("armature", str(reduced_inertia))

        motor = root.find(f".//motor[@joint='{joint.get('name')}']")
        if motor is not None:
            motor.set("ctrlrange", f"-{drive['max_torque']} {drive['max_torque']}")

    variant_file = f"ur3_{drive['model']}.xml"
    tree.write(variant_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    variant_files.append(variant_file)
