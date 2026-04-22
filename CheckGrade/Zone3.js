import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ScrollView, ActivityIndicator, SafeAreaView, Platform, TextInput } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

// --- ZONE 3 SPECIFIC CHECKLIST DATA ---
const ZONE_3_CHECKLIST = [
  { id: 'z3_1', title: 'Equipment & Critical Care', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
  { id: 'z3_2', title: 'Medicines & Records', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
  { id: 'z3_3', title: 'Rest Area', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
  { id: 'z3_4', title: 'Feeding Room & Kitchen', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
  { id: 'z3_5', title: 'Creche – Bed & Play Area', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
  { id: 'z3_6', title: 'Ambulance', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
  { id: 'z3_7', title: 'Waste Segregation', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
  { id: 'z3_8', title: 'Cleaning Standards', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
];

export default function Zone3({ goBack }) {
  const [auditorName, setAuditorName] = useState('');
  
  const currentDate = new Date();
  const formattedDate = currentDate.toLocaleDateString();
  const formattedTime = currentDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const currentMonth = currentDate.toLocaleString('default', { month: 'long' });

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.auditHeader}>
        <TouchableOpacity style={styles.backButton} onPress={goBack}>
          <Text style={styles.backButtonText}>← Dashboard</Text>
        </TouchableOpacity>
        <Text style={styles.auditHeaderTitle}>Zone 3</Text>
        <Text style={styles.auditHeaderBadge}>CRÈCHE AND MEDICAL ROOM</Text>
      </View>
      
      <ScrollView contentContainerStyle={styles.checklistContainer} showsVerticalScrollIndicator={false}>
        
        {/* --- AUDIT DETAILS CARD --- */}
        <View style={styles.metaCard}>
          <Text style={styles.metaCardTitle}>Audit Details</Text>
          <View style={styles.metaRow}>
            <Text style={styles.metaLabel}>Auditor Name</Text>
            <TextInput 
              style={styles.metaInput} 
              placeholder="Enter auditor's name..." 
              value={auditorName} 
              onChangeText={setAuditorName}
              placeholderTextColor="#9CA3AF"
            />
          </View>
          <View style={styles.metaRowSplit}>
            <View style={styles.metaHalf}>
              <Text style={styles.metaLabel}>Date & Time</Text>
              <Text style={styles.metaValueText}>{formattedDate} - {formattedTime}</Text>
            </View>
            <View style={styles.metaHalf}>
              <Text style={styles.metaLabel}>Audit Month</Text>
              <Text style={styles.metaValueText}>{currentMonth}</Text>
            </View>
          </View>
        </View>

        {/* --- CHECKLIST ITEMS --- */}
        {ZONE_3_CHECKLIST.map((item, index) => (
          <ChecklistCard key={item.id} point={item} index={index + 1} />
        ))}
        
        <TouchableOpacity style={styles.submitZoneButton} onPress={goBack}>
          <Text style={styles.submitZoneText}>Complete Zone 3 Audit</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

function ChecklistCard({ point, index }) {
  const [actualImage, setActualImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiScore, setAiScore] = useState(null);

  const takePicture = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 });
    if (!result.canceled) { setActualImage(result.assets[0].uri); setAiScore(null); }
  };

  const analyzeImages = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setAiScore((Math.random() * 2 + 3).toFixed(1)); 
      setIsAnalyzing(false);
    }, 2000);
  };

  return (
    <View style={styles.checklistCard}>
      <View style={styles.checklistCardHeader}>
        <View style={styles.indexCircle}><Text style={styles.indexText}>{index}</Text></View>
        <Text style={styles.checklistTitle}>{point.title}</Text>
      </View>

      <View style={styles.comparisonRow}>
        <View style={styles.imageCol}>
          <Text style={styles.imageLabel}>Standard</Text>
          <Image source={{ uri: point.standardImg }} style={styles.compareImage} />
        </View>
        <View style={styles.imageCol}>
          <Text style={styles.imageLabel}>Actual</Text>
          {actualImage ? (
            <Image source={{ uri: actualImage }} style={styles.compareImage} />
          ) : (
            <TouchableOpacity style={styles.comparePlaceholder} onPress={takePicture}>
              <Text style={styles.placeholderIcon}>📷</Text>
              <Text style={styles.placeholderTapText}>Tap to Scan</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      <View style={styles.actionRow}>
        {actualImage && !aiScore && !isAnalyzing && (
          <TouchableOpacity style={styles.aiButton} onPress={analyzeImages}><Text style={styles.aiButtonText}>Run AI Vision Analysis</Text></TouchableOpacity>
        )}
        {isAnalyzing && (
          <View style={styles.analyzingBox}><ActivityIndicator size="small" color="#6366F1" /><Text style={styles.analyzingText}>Grading...</Text></View>
        )}
        {aiScore && (
          <View style={styles.scoreResult}><Text style={styles.scoreResultLabel}>AI Grade:</Text><Text style={styles.scoreResultValue}>{aiScore}/5</Text></View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F3F4F6' },
  auditHeader: { paddingHorizontal: 20, paddingTop: Platform.OS === 'android' ? 50 : 20, paddingBottom: 20, backgroundColor: '#FFF', borderBottomWidth: 1, borderColor: '#E5E7EB' },
  backButton: { marginBottom: 15 },
  backButtonText: { fontSize: 16, color: '#6366F1', fontWeight: '700' },
  auditHeaderTitle: { fontSize: 28, fontWeight: '900', color: '#111827' },
  auditHeaderBadge: { fontSize: 14, color: '#6B7280', fontWeight: '700', marginTop: 5, textTransform: 'uppercase' },
  checklistContainer: { padding: 15, paddingBottom: 50, maxWidth: 1000, width: '100%', alignSelf: 'center' },
  
  metaCard: { backgroundColor: '#FFF', borderRadius: 16, padding: 20, marginBottom: 25, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3, borderWidth: 1, borderColor: '#E5E7EB' },
  metaCardTitle: { fontSize: 18, fontWeight: '800', color: '#111827', marginBottom: 15 },
  metaRow: { marginBottom: 15 },
  metaRowSplit: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 5 },
  metaHalf: { flex: 1 },
  metaLabel: { fontSize: 12, fontWeight: '700', color: '#6B7280', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 },
  metaInput: { backgroundColor: '#F9FAFB', borderWidth: 1, borderColor: '#D1D5DB', borderRadius: 8, paddingHorizontal: 15, paddingVertical: 12, fontSize: 16, color: '#111827', fontWeight: '500' },
  metaValueText: { fontSize: 16, fontWeight: '600', color: '#111827' },

  checklistCard: { backgroundColor: '#FFF', borderRadius: 16, padding: 15, marginBottom: 20, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.05, shadowRadius: 8, elevation: 3 },
  checklistCardHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 15 },
  indexCircle: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#EFF6FF', justifyContent: 'center', alignItems: 'center', marginRight: 12 },
  indexText: { color: '#3B82F6', fontWeight: 'bold', fontSize: 16 },
  checklistTitle: { fontSize: 18, fontWeight: '800', color: '#1F2937', flex: 1 },
  comparisonRow: { flexDirection: 'row', gap: 10, marginBottom: 15 },
  imageCol: { flex: 1 },
  imageLabel: { fontSize: 12, fontWeight: '700', color: '#6B7280', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.5 },
  compareImage: { width: '100%', height: 140, borderRadius: 12, resizeMode: 'cover', backgroundColor: '#F3F4F6' },
  comparePlaceholder: { width: '100%', height: 140, borderRadius: 12, backgroundColor: '#F9FAFB', borderWidth: 2, borderColor: '#E5E7EB', borderStyle: 'dashed', justifyContent: 'center', alignItems: 'center' },
  placeholderIcon: { fontSize: 24, marginBottom: 5 },
  placeholderTapText: { color: '#9CA3AF', fontSize: 12, fontWeight: '600' },
  actionRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 5 },
  aiButton: { backgroundColor: '#111827', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 10, flex: 1, alignItems: 'center', marginRight: 10 },
  aiButtonText: { color: '#FFF', fontWeight: '700', fontSize: 14 },
  analyzingBox: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#EEF2FF', paddingHorizontal: 15, paddingVertical: 10, borderRadius: 10, flex: 1, marginRight: 10 },
  analyzingText: { marginLeft: 10, color: '#4F46E5', fontWeight: '600', fontSize: 14 },
  scoreResult: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F0FDF4', paddingHorizontal: 15, paddingVertical: 10, borderRadius: 10, flex: 1, marginRight: 10, borderWidth: 1, borderColor: '#BBF7D0' },
  scoreResultLabel: { color: '#166534', fontWeight: '700', fontSize: 14, marginRight: 10 },
  scoreResultValue: { color: '#15803D', fontWeight: '900', fontSize: 18 },
  submitZoneButton: { backgroundColor: '#10B981', padding: 20, borderRadius: 16, alignItems: 'center', marginTop: 10, marginBottom: 40 },
  submitZoneText: { color: '#FFF', fontSize: 18, fontWeight: '800', letterSpacing: 0.5 }
});