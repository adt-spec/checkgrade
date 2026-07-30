import os
import re
import glob

zone_files = glob.glob('CheckGrade/*.js')

for file_path in zone_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Ensure Alert and Platform are imported from react-native
    rn_import_match = re.search(r'import\s+{([^}]+)}\s+from\s+[\'"]react-native[\'"]', content)
    if rn_import_match:
        imports = [i.strip() for i in rn_import_match.group(1).split(',')]
        if 'Alert' not in imports:
            imports.append('Alert')
        if 'Platform' not in imports:
            imports.append('Platform')
        
        new_import = f"import {{ {', '.join(imports)} }} from 'react-native'"
        content = content[:rn_import_match.start()] + new_import + content[rn_import_match.end():]

    # 2. Find and replace takePicture
    pattern = r'const takePicture = async \(\) => \{[\s\S]*?launchImageLibraryAsync[\s\S]*?if \(!result\.canceled\) \{[\s\S]*?setActualImage\(result\.assets\[0\]\.uri\);\s*(setAiScore\(null\);|setAiResult\(null\);)[\s\S]*?\}[\s\S]*?\};'
    
    match = re.search(pattern, content)
    if match:
        set_state_cmd = match.group(1) # e.g. setAiScore(null); or setAiResult(null);
        
        new_take_picture = f"""const takePicture = () => {{
    if (Platform.OS === 'web') {{
      ImagePicker.launchImageLibraryAsync({{ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 }}).then(result => {{
        if (!result.canceled) {{ setActualImage(result.assets[0].uri); {set_state_cmd} }}
      }});
      return;
    }}
    Alert.alert(
      "Upload Image",
      "Choose an option",
      [
        {{ text: "Camera", onPress: async () => {{
          await ImagePicker.requestCameraPermissionsAsync();
          const result = await ImagePicker.launchCameraAsync({{ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 }});
          if (!result.canceled) {{ setActualImage(result.assets[0].uri); {set_state_cmd} }}
        }}}},
        {{ text: "Gallery", onPress: async () => {{
          const result = await ImagePicker.launchImageLibraryAsync({{ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 }});
          if (!result.canceled) {{ setActualImage(result.assets[0].uri); {set_state_cmd} }}
        }}}},
        {{ text: "Cancel", style: "cancel" }}
      ]
    );
  }};"""
        
        content = content[:match.start()] + new_take_picture + content[match.end():]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {file_path}")
    else:
        print(f"Skipped {file_path} (could not match takePicture)")
