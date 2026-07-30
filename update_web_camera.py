import glob
import re

zone_files = glob.glob('CheckGrade/*.js')

for file_path in zone_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to replace the `if (Platform.OS === 'web') { ... }` block inside takePicture.
    # We will search for this exact pattern block.
    
    pattern = r"if \(Platform\.OS === 'web'\) \{[\s\S]*?ImagePicker\.launchImageLibraryAsync[\s\S]*?then\(result => \{[\s\S]*?if \(!result\.canceled\) \{ setActualImage\(result\.assets\[0\]\.uri\);\s*(setAiScore\(null\);|setAiResult\(null\);) \}[\s\S]*?\}\);[\s\S]*?return;[\s\S]*?\}"
    
    match = re.search(pattern, content)
    
    if match:
        set_state_cmd = match.group(1) # setAiScore(null); or setAiResult(null);
        
        new_web_block = f"""if (Platform.OS === 'web') {{
      const useCamera = window.confirm("Use Camera? \\n(Click 'OK' for Camera, 'Cancel' for Gallery)");
      if (useCamera) {{
        ImagePicker.launchCameraAsync({{ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 }}).then(result => {{
          if (!result.canceled) {{ setActualImage(result.assets[0].uri); {set_state_cmd} }}
        }});
      }} else {{
        ImagePicker.launchImageLibraryAsync({{ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 }}).then(result => {{
          if (!result.canceled) {{ setActualImage(result.assets[0].uri); {set_state_cmd} }}
        }});
      }}
      return;
    }}"""
        
        content = content[:match.start()] + new_web_block + content[match.end():]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {file_path}")
    else:
        print(f"Skipped {file_path} (could not find web block)")
