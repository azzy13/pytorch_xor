from setuptools import setup

package_name = "pytorch_xor"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hasana3",
    maintainer_email="azhar.hasan@vanderbilt.edu",
    description="Basic Pub-Sub with pytorch_xor",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "torch = pytorch_xor.demo_pytorch:main",
            "talker = pytorch_xor.publisher_member_function:main",
            "listener = pytorch_xor.subscriber_member_function:main",
        ],
    },
)
