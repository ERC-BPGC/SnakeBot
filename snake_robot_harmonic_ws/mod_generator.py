import xml.etree.ElementTree as ET

# Define the module template with placeholders
module_template = '''
<body name="base_link_{index}" pos="{pos}" {quat}>
    {joint}
    <inertial pos="0 0 0" mass="0.1" diaginertia="0.1 0.1 0.1"/>
    <geom type="mesh" contype="1" conaffinity="1" group="1" density="0" rgba="0.752941 0.752941 0.752941 1" mesh="base_link"/>
    <geom type="capsule" size="0.04 0.02" pos="0 0 0" rgba="0.752941 0.752941 0.752941 0" contype="0" conaffinity="0"/>
    <body name="rotary_connector_1_{index}" pos="0.04 -0.017 -1.4731e-05" quat="0.499998 0.5 -0.5 0.500002">
        <inertial pos="5.23676e-08 0.00591355 0.000283844" quat="0.707096 0.00392812 0.00392812 0.707096" mass="0.00371393" diaginertia="2.62338e-07 1.57392e-07 1.5609e-07"/>
        <joint name="rotary_connector_1_{index}_joint" pos="0 0 0" axis="0 -1 0"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.627451 0.627451 0.627451 1" mesh="rotary_connector_1"/>
        <body name="rod_1_{index}" pos="0 0.005 0.0125" quat="0.707105 0 0 -0.707108">
            <inertial pos="-0.0350209 0.0158107 0.0275138" quat="-0.000114745 0.866198 0.000304169 0.499701" mass="0.0120182" diaginertia="1.50991e-05 1.28905e-05 2.65306e-06"/>
            <joint name="rod_1_{index}_joint" pos="0 0 0" axis="0 -1 0.00018414"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.627451 0.627451 0.627451 1" mesh="rod_1"/>
            <geom type="capsule" size="0.01 0.035" pos="-0.035 0.015 0.0275" quat="0.707 0 0.707 0" rgba="0.627451 0.627451 0.627451 0" contype="0" conaffinity="0"/>
            <body name="rotary_connector_4_{index}" pos="-0.07 1.0128e-05 0.055" quat="-3.67321e-06 0 0 1">
                <site name="site_rotary_connector_4_{index}" type="sphere" size="0.0015" pos="0 0 0.0125" rgba="1 0 0 1"/>
                <inertial pos="8.37506e-11 -2.23599e-06 0.012143" mass="0.0029582" diaginertia="2.22521e-07 1.36818e-07 1.35501e-07"/>
                <joint name="rotary_connector_4_{index}_joint" pos="0 0 0" axis="0 -1 -0.00018414"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.627451 0.627451 0.627451 1" mesh="rotary_connector_4"/>
            </body>
        </body>
    </body>
    <body name="rotary_connector_2_{index}" pos="0 -0.017 -0.04" quat="0.707105 0.707108 0 0">
        <inertial pos="-6.5746e-08 0.005 0.00035704" quat="0.707107 0 0 0.707107" mass="0.0029582" diaginertia="2.2252e-07 1.3682e-07 1.355e-07"/>
        <joint name="rotary_connector_2_{index}_joint" pos="0 0 0" axis="0 -1 0"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.62745 0.62745 0.62745 1" mesh="rotary_connector_2"/>
        <body name="rod_2_{index}" pos="0 0.005 0.0125" quat="0.707105 0 0 -0.707108">
            <inertial pos="-0.0350209 0.0158006 0.0275196" quat="-0.00027424 0.866198 0.000212159 0.499701" mass="0.0120182" diaginertia="1.50991e-05 1.28905e-05 2.65306e-06"/>
            <joint name="rod_2_{index}_joint" pos="0 0 0" axis="0 -1 -0.00018414"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.627451 0.627451 0.627451 1" mesh="rod_2"/>
            <geom type="capsule" size="0.01 0.035" pos="-0.035 0.015 0.0275" quat="0.707 0 0.707 0" rgba="0.627451 0.627451 0.627451 0" contype="0" conaffinity="0"/>
            <body name="rotary_connector_5_{index}" pos="-0.07 -1.0128e-05 0.055" quat="-3.67321e-06 0 0 1">
                <inertial pos="0 2.236e-06 0.012143" mass="0.0029582" diaginertia="2.2252e-07 1.3682e-07 1.355e-07"/>
                <joint name="rotary_connector_5_{index}_joint" pos="0 0 0" axis="0 -1 0.00018414"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.62745 0.62745 0.62745 1" mesh="rotary_connector_5"/>
                <body name="top_base_{index}" pos="-0.035 0.04 0.0125" quat="0 0 0 1">
                    <inertial pos="-0.00316689 0.0389291 0.00837195" quat="-0.0188662 0.70606 0.0239272 0.707496" mass="0.100446" diaginertia="0.000104954 5.73207e-05 5.19343e-05"/>
                    <site name="site_1_top_base_{index}" type="sphere" size="0.0015" pos="0.0 0.005 0.0" rgba="1 0 0 1"/>
                    <site name="site_2_top_base_{index}" type="sphere" size="0.0025" pos="0.0315 0.0725 0.0" rgba="0 1 0 1"/>
                    <site name="site_3_top_base_{index}" type="sphere" size="0.0015" pos="-0.035 0.04 0" rgba="0 1 1 1"/>
                    <joint name="top_base_{index}_joint" pos="-0.035 0.04 0" axis="1 0 0"/>
                    <geom type="mesh" contype="1" conaffinity="1" group="1" rgba="0.627451 0.627451 0.627451 1" mesh="top_base"/>
                    <geom type="box" size="0.02 0.04 0.01" pos="0 0.04 0.01" rgba="0.627451 0.627451 0.627451 0" contype="0" conaffinity="0"/>
                </body>
            </body>
        </body>
    </body>
    <body name="rotary_connector_3_{index}" pos="-0.035355 -0.017 0.035355" quat="-0.270602 -0.270596 -0.653283 0.653279">
        <inertial pos="0 0.005 0.000357044" quat="0.707107 0 0 0.707107" mass="0.0029582" diaginertia="2.22521e-07 1.36818e-07 1.35501e-07"/>
        <joint name="rotary_connector_3_{index}_joint" pos="0 0 0" axis="0 -1 0"/>
        <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.627451 0.627451 0.627451 1" mesh="rotary_connector_3"/>
        <body name="support_rod_{index}" pos="0 0.005 0.0125" quat="0.707105 0 0 -0.707108">
            <inertial pos="-0.0450011 0.0218276 0.0274999" quat="-4.62052e-06 0.836519 1.4047e-06 0.547938" mass="0.0144478" diaginertia="2.60758e-05 2.18241e-05 4.7952e-06"/>
            <joint name="support_rod_{index}_joint" pos="0 0 0" axis="0 -1 0"/>
            <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0 0 0 1" mesh="support_rod"/>
            <geom type="capsule" size="0.01 0.045" pos="-0.045 0.02 0.0275" quat="0.707 0 0.707 0" rgba="0.627451 0.627451 0.627451 0" contype="0" conaffinity="0"/>
            <body name="rotary_connector_6_{index}" pos="-0.09 0 0.055" quat="-3.67321e-06 0 0 1">
                <site name="site_rotary_connector_6_{index}" type="sphere" size="0.0015" pos="0 0 0.0125" rgba="0 1 0 1"/>
                <inertial pos="-1.18443e-10 0 0.012143" mass="0.0029582" diaginertia="2.22521e-07 1.36818e-07 1.35501e-07"/>
                <joint name="rotary_connector_6_{index}_joint" pos="0 0 0" axis="0 -1 0"/>
                <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.627451 0.627451 0.627451 1" mesh="rotary_connector_6"/>
            </body>
        </body>
    </body>
</body>
'''

# Function to generate XML for a single module
def generate_module_xml(index, is_first=False):
    if is_first:
        pos = "0 0 0.1"
        quat = ""
        joint = '<joint type="free"/>'
    else:
        pos = "0 0.04 0.0175"
        quat = 'quat="0.5 -0.5 -0.5 0.5"'
        joint = ""
    return module_template.format(index=index, pos=pos, quat=quat, joint=joint)

# Function to build the SnakeBot with multiple modules
def build_snakebot(num_modules):
    if num_modules < 1:
        return ""
    
    # Generate all modules
    modules = [generate_module_xml(i, i == 1) for i in range(1, num_modules + 1)]
    
    # Parse the first module as the root
    root = ET.fromstring(modules[0])
    
    # Nest subsequent modules
    if num_modules > 1:
        current_top_base = root.find(".//body[@name='top_base_1']")
        for i in range(1, num_modules):
            next_module = ET.fromstring(modules[i])
            current_top_base.append(next_module)
            current_top_base = next_module.find(f".//body[@name='top_base_{i+1}']")
    
    # Convert back to string
    return ET.tostring(root, encoding='unicode')

# Example usage: Generate XML for 3 modules and save to file
if __name__ == "__main__":
    xml_content = build_snakebot(3)
    with open("snakebot.xml", "w") as f:
        f.write('<mujoco>\n<worldbody>\n' + xml_content + '\n</worldbody>\n</mujoco>')
    print("SnakeBot XML with 3 modules generated and saved to 'snakebot.xml'")