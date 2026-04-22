import React, { useState } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, SafeAreaView, KeyboardAvoidingView, Platform, Image, useWindowDimensions } from 'react-native';
import { Feather } from '@expo/vector-icons';

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const { width } = useWindowDimensions();

  // If the screen is less than 850px wide, switch to mobile/tablet column layout
  const isMobile = width < 850;

  const handleLogin = () => {
    if (email && password) {
      onLogin(); 
    } else {
      alert("Please enter both an Email/ID and Password.");
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.container}>
        
        {/* Main Responsive Card */}
        <View style={[styles.card, isMobile && styles.cardMobile]}>
          
          {/* ========================================== */}
          {/* LEFT SIDE: FORM PANEL */}
          {/* ========================================== */}
          <View style={[styles.leftPanel, isMobile && styles.leftPanelMobile]}>
            <View style={styles.formContent}>
              
              <View style={styles.pillBadge}>
                <Text style={styles.pillBadgeText}>LOGIN PAGE</Text>
              </View>

              <View style={styles.inputWrapper}>
                <Feather name="user" size={18} color="#A0AEC0" />
                <TextInput
                  style={styles.input}
                  placeholder="e.g. auditor@lagunaclothing.com"
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                  placeholderTextColor="#A0AEC0"
                />
              </View>

              <View style={styles.inputWrapper}>
                <Feather name="lock" size={18} color="#A0AEC0" />
                <TextInput
                  style={styles.input}
                  placeholder="Enter your password"
                  value={password}
                  onChangeText={setPassword}
                  secureTextEntry={!showPassword}
                  placeholderTextColor="#A0AEC0"
                />
                <TouchableOpacity onPress={() => setShowPassword(!showPassword)}>
                  <Feather name={showPassword ? "eye" : "eye-off"} size={18} color="#A0AEC0" />
                </TouchableOpacity>
              </View>

              <TouchableOpacity style={styles.loginButton} onPress={handleLogin} activeOpacity={0.85}>
                <Text style={styles.loginButtonText}>LOGIN</Text>
                <Feather name="arrow-right" size={18} color="#FFFFFF" style={{ marginLeft: 8 }} />
              </TouchableOpacity>
              
            </View>
          </View>

          {/* ========================================== */}
          {/* RIGHT SIDE: BRANDING PANEL */}
          {/* ========================================== */}
          <View style={[styles.rightPanel, isMobile && styles.rightPanelMobile]}>
            <View style={styles.brandingContent}>
              
              <Text style={styles.brandTitle}>
                CHECK<Text style={styles.brandSubtitle}>GRADE</Text>
              </Text>

              <Image 
                source={require('./assets/logo.png')} 
                style={styles.logo} 
              />

            </View>
          </View>

        </View>

        {/* Footer Text */}
        <Text style={styles.footerText}>ADT © LAGUNA CLOTHING PVT LTD</Text>
        
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// ==========================================
// --- TRULY RESPONSIVE STYLESHEET ---
// ==========================================
const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F4F6F8' }, 
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: '3%' },
  
  // The Main White Card
  card: { 
    flexDirection: 'row', 
    backgroundColor: '#FFFFFF', 
    borderRadius: 24, 
    width: '100%', 
    maxWidth: 1000, 
    minHeight: 550, // Replaced fixed height with minHeight so it can grow or shrink naturally
    elevation: 8, 
    shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.08, shadowRadius: 20 
  },
  cardMobile: { 
    flexDirection: 'column-reverse', // Stacks the branding on top of the form on small screens
    minHeight: 'auto', 
    maxWidth: 450 // Limits width so it doesn't look stretched on tablets
  },

  // Panels
  leftPanel: { 
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center', 
    padding: '5%', // Percentage padding ensures it shrinks on smaller windows
    backgroundColor: '#FFFFFF',
    borderBottomLeftRadius: 24,
    borderTopLeftRadius: 24,
  },
  leftPanelMobile: { paddingVertical: 40, borderTopLeftRadius: 0, borderBottomRightRadius: 24 },
  
  formContent: { width: '100%', maxWidth: 350, alignItems: 'center' },

  rightPanel: { 
    flex: 1, 
    justifyContent: 'center',
    alignItems: 'center', 
    padding: '5%', 
    borderLeftWidth: 1, 
    borderColor: '#F0F0F0', 
    borderStyle: 'dashed' 
  },
  rightPanelMobile: { borderLeftWidth: 0, borderBottomWidth: 1, paddingVertical: 40 },
  
  brandingContent: { 
    width: '100%', 
    alignItems: 'center', 
    justifyContent: 'center' 
  },

  // Logo & Branding
  brandTitle: { 
    fontSize: 36, 
    fontWeight: '900', 
    color: '#000000', 
    letterSpacing: 1, 
    marginBottom: '10%', // Responsive margin
    textAlign: 'center'
  },
  brandSubtitle: { color: '#8E96A4', fontWeight: '300' },
  
  // Responsive Logo: Scales perfectly using aspect ratio instead of hard limits
  logo: { 
    width: '100%', 
    maxWidth: 320, 
    aspectRatio: 1, // Keeps the image perfectly square without squishing
    resizeMode: 'contain' 
  }, 

  // Badges
  pillBadge: { backgroundColor: '#F1F5F9', paddingHorizontal: 20, paddingVertical: 8, borderRadius: 20, marginBottom: 40 },
  pillBadgeText: { color: '#64748B', fontSize: 11, fontWeight: '700', letterSpacing: 1 },

  // Input Fields
  inputWrapper: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    backgroundColor: '#F3F6F9', 
    borderRadius: 25, 
    paddingHorizontal: 20, 
    height: 55, 
    marginBottom: 20, 
    width: '100%' 
  },
  input: { 
    flex: 1, 
    marginLeft: 15, 
    color: '#1F2937', 
    fontSize: 15, 
    fontWeight: '500',
    ...(Platform.OS === 'web' && { outlineStyle: 'none' }) 
  },

  // Login Button
  loginButton: { 
    flexDirection: 'row',
    backgroundColor: '#000000', 
    borderRadius: 25, 
    height: 55, 
    width: '100%',
    alignItems: 'center', 
    justifyContent: 'center',
    marginTop: 10,
    elevation: 3,
    shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.15, shadowRadius: 6
  },
  loginButtonText: { color: '#FFFFFF', fontSize: 14, fontWeight: '800', letterSpacing: 1.5 },

  // Footer
  footerText: { marginTop: 30, color: '#A0AEC0', fontSize: 10, fontWeight: '700', letterSpacing: 1, textAlign: 'center' }
});