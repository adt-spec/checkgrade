import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ScrollView, ActivityIndicator, SafeAreaView, Platform, TextInput, Modal, KeyboardAvoidingView } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Feather, MaterialIcons } from '@expo/vector-icons';
import { Asset } from 'expo-asset';

const ZONE_1_CHECKLIST = [
  { id: 'c1', title: 'Security Desk', standardImg: require('../assets/Zone 1.png') },
  { id: 'c2', title: 'Item Storage', standardImg: require('../assets/Zone1_Storage.jpeg') },
  { id: 'c3', title: 'Visitor Record Keeping', standardImg: require('../assets/Zone1_Visitor.jpeg') },
  { id: 'c4', title: 'Electrical Eqp. & Control Panel', standardImg: require('../assets/Zone1_Electrical.jpeg') },
  { id: 'c5', title: 'Baggage Check', standardImg: require('../assets/Zone1_Baggage.jpeg') },
  { id: 'c6', title: 'Truck Insp. & Material Movement', standardImg: require('../assets/Zone1_Truck.jpeg') },
  { id: 'c7', title: 'Cleaning Standards', standardImg: require('../assets/Zone1_Cleaning.jpeg') },
];

// MATERIAL DESIGN COLOR PALETTE
const COLORS = {
  primary: '#0F766E',
  primaryLight: '#CCFBF1',
  secondary: '#FF6F00',
  surface: '#FFFFFF',
  background: '#F1F5F9',
  textHigh: '#0F172A',
  textMedium: '#475569',
  textLow: '#94A3B8',
  error: '#E11D48',
  errorLight: '#FFE4E6',
  success: '#10B981',
  successLight: '#D1FAE5',
  warning: '#F59E0B',
  warningLight: '#FEF3C7',
  border: '#E2E8F0'
};

export default function Zone1({ goBack, onComplete, auditHistory }) {
  const [auditorName, setAuditorName] = useState('');
  const [zoneDataPayload, setZoneDataPayload] = useState({});
  const [modalVisible, setModalVisible] = useState(false);
  const [modalConfig, setModalConfig] = useState({ title: '', message: '', type: 'success', onConfirm: null });
  const [vaultModalVisible, setVaultModalVisible] = useState(false);

  const currentDate = new Date();
  const formattedDate = currentDate.toLocaleDateString();
  const formattedTime = currentDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  const zoneRecords = (auditHistory || []).filter(r => r.zoneId === '1');
  const completedCount = Object.keys(zoneDataPayload).length;
  const totalCount = ZONE_1_CHECKLIST.length;
  const progressPercent = (completedCount / totalCount) * 100;

  const handleScoreUpdate = (id, fullResultData) => {
    setZoneDataPayload(prev => ({ ...prev, [id]: fullResultData }));
  };

  const calculateAverage = () => {
    const items = Object.values(zoneDataPayload);
    if (items.length === 0) return "0.0";
    const total = items.reduce((sum, item) => sum + item.score, 0);
    return (total / items.length).toFixed(1);
  };

  const showModal = (title, message, type, onConfirm) => {
    setModalConfig({ title, message, type, onConfirm });
    setModalVisible(true);
  };

  const handleCompleteAudit = () => {
    if (completedCount < totalCount) {
      showModal("Audit Incomplete", `You have only graded ${completedCount} out of ${totalCount} items. Please complete all checkpoints.`, "warning", null);
      return;
    }
    const finalAverage = calculateAverage();
    showModal(
      "Audit Complete",
      `Auditor: ${auditorName || 'Not Provided'}\nFinal Score: ${finalAverage} / 5.0\n\nThe report has been securely saved to the vault.`,
      "success",
      () => {
        setTimeout(() => {
          if (onComplete) onComplete(finalAverage, zoneDataPayload, auditorName);
          else if (goBack) goBack();
        }, 100);
      }
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.heroBackground} />

      <View style={styles.appBar}>
        <View style={styles.appBarLeft}>
          <TouchableOpacity style={styles.iconButton} onPress={goBack}>
            <MaterialIcons name="arrow-back" size={26} color={COLORS.surface} />
          </TouchableOpacity>
          <View>
            <Text style={styles.appBarSubtitle}>SECURITY</Text>
            <Text style={styles.appBarTitle}>Zone 1 Audit</Text>
          </View>
        </View>

        <TouchableOpacity 
            style={[styles.historyFab, zoneRecords.length === 0 && { backgroundColor: 'transparent', borderColor: 'rgba(255,255,255,0.2)' }]} 
            onPress={() => zoneRecords.length > 0 ? setVaultModalVisible(true) : showModal("No History", "There are no past audits for this zone yet.", "warning", null)}
            activeOpacity={zoneRecords.length === 0 ? 1 : 0.8}
        >
            <MaterialIcons name="history" size={20} color={zoneRecords.length === 0 ? 'rgba(255,255,255,0.5)' : COLORS.surface} />
            <Text style={[styles.historyFabText, zoneRecords.length === 0 && { color: 'rgba(255,255,255,0.5)' }]}>Vault ({zoneRecords.length})</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.progressContainer}>
        <View style={styles.progressBarBg}>
            <View style={[styles.progressBarFill, { width: `${progressPercent}%` }]} />
        </View>
        <Text style={styles.progressText}>{completedCount} of {totalCount} Checkpoints Completed</Text>
      </View>

      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={{ flex: 1 }}>
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          <View style={styles.formSheet}>
              <View style={styles.formSection}>
                <Text style={styles.sectionTitle}>Audit</Text>
                
                <View style={styles.materialInputContainer}>
                    <MaterialIcons name="person" size={20} color={COLORS.textMedium} style={styles.inputIcon} />
                    <View style={styles.inputWrapper}>
                        <Text style={styles.inputLabel}>Auditor Name</Text>
                        <TextInput 
                            style={styles.materialInput} 
                            placeholder="Enter name..." 
                            value={auditorName} 
                            onChangeText={setAuditorName} 
                            placeholderTextColor={COLORS.textLow} 
                        />
                    </View>
                </View>

                <View style={styles.metaChipsRow}>
                    <View style={styles.metaChip}>
                        <MaterialIcons name="event" size={16} color={COLORS.primary}/>
                        <Text style={styles.metaChipText}>{formattedDate}</Text>
                    </View>
                    <View style={styles.metaChip}>
                        <MaterialIcons name="schedule" size={16} color={COLORS.primary}/>
                        <Text style={styles.metaChipText}>{formattedTime}</Text>
                    </View>
                </View>
              </View>

              {ZONE_1_CHECKLIST.map((item, index) => (
                <ChecklistSection
                  key={item.id}
                  point={item}
                  index={index + 1}
                  isLast={index === ZONE_1_CHECKLIST.length - 1}
                  onResult={(resultData) => handleScoreUpdate(item.id, resultData)} 
                  onError={(errMsg) => showModal("Analysis Error", errMsg, "error", null)}
                />
              ))}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      <View style={styles.bottomAppBar}>
          <View style={styles.scoreDisplay}>
              <Text style={styles.scoreDisplayLabel}>Current Avg</Text>
              <Text style={styles.scoreDisplayValue}>{calculateAverage()}</Text>
          </View>
          <TouchableOpacity 
            style={[styles.fabExtended, completedCount < totalCount && styles.fabDisabled]} 
            onPress={handleCompleteAudit} 
            activeOpacity={0.8}
          >
              <MaterialIcons name="done-all" size={22} color={COLORS.surface} style={{marginRight: 8}} />
              <Text style={styles.fabExtendedText}>Complete Audit</Text>
          </TouchableOpacity>
      </View>

      <Modal transparent={true} visible={modalVisible} animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.dialogCard}>
            <View style={[styles.dialogIconBox, modalConfig.type === 'success' ? { backgroundColor: COLORS.successLight } : modalConfig.type === 'warning' ? { backgroundColor: COLORS.warningLight } : { backgroundColor: COLORS.errorLight }]}>
              <MaterialIcons name={modalConfig.type === 'success' ? "check-circle" : modalConfig.type === 'warning' ? "warning" : "error"} size={36} color={modalConfig.type === 'success' ? COLORS.success : modalConfig.type === 'warning' ? COLORS.warning : COLORS.error} />
            </View>
            <Text style={styles.dialogTitle}>{modalConfig.title}</Text>
            <Text style={styles.dialogMessage}>{modalConfig.message}</Text>
            <TouchableOpacity style={styles.dialogButton} activeOpacity={0.8} onPress={() => { setModalVisible(false); if (modalConfig.onConfirm) modalConfig.onConfirm(); }}>
              <Text style={styles.dialogButtonText}>Okay</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <Modal transparent={true} visible={vaultModalVisible} animationType="slide">
        <View style={styles.vaultModalOverlay}>
          <View style={styles.vaultModalCard}>
            <View style={styles.vaultModalHeader}>
                <View style={{flex: 1}}>
                    <Text style={styles.vaultModalBadge}>SECURITY VAULT</Text>
                    <Text style={styles.vaultModalTitle}>Zone 1 History</Text>
                </View>
                <TouchableOpacity onPress={() => setVaultModalVisible(false)} style={styles.iconButtonDark}>
                    <MaterialIcons name="close" size={24} color={COLORS.textHigh} />
                </TouchableOpacity>
            </View>
            <ScrollView style={styles.vaultModalScroll} showsVerticalScrollIndicator={false}>
                {zoneRecords.map((record, index) => (
                    <View key={record.id} style={styles.timelineRow}>
                        <View style={styles.timelineDotBox}>
                            <View style={[styles.timelineDot, { backgroundColor: record.score >= 4 ? COLORS.success : record.score >= 2.5 ? COLORS.warning : COLORS.error }]} />
                            {index !== zoneRecords.length - 1 && <View style={styles.timelineLine} />}
                        </View>
                        <View style={styles.timelineContent}>
                            <View style={styles.timelineTop}>
                                <Text style={styles.timelineDate}>{record.date}</Text>
                                <View style={[styles.timelineScoreBadge, { backgroundColor: record.score >= 4 ? COLORS.successLight : record.score >= 2.5 ? COLORS.warningLight : COLORS.errorLight }]}>
                                    <Text style={[styles.timelineScoreText, { color: record.score >= 4 ? COLORS.success : record.score >= 2.5 ? COLORS.warning : COLORS.error }]}>{record.score} / 5.0</Text>
                                </View>
                            </View>
                            <Text style={styles.timelineAuditor}>Auditor: {record.auditor}</Text>
                        </View>
                    </View>
                ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

function ChecklistSection({ point, index, isLast, onResult, onError }) {
  const [actualImage, setActualImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const selectedEngine = 'gemini'; 

  const takePicture = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.7 });
    if (!result.canceled) {
      setActualImage(result.assets[0].uri);
      setAiResult(null);
    }
  };

  const analyzeImages = async () => {
    setIsAnalyzing(true);
    setAiResult(null);
    
    console.log(`[CHECKPOINT ${index}] Starting analysis...`);
    
    try {
      let formData = new FormData();
      let finalStandardUrl = point.standardImg;

      if (typeof finalStandardUrl === 'object' && finalStandardUrl !== null && finalStandardUrl.uri) {
          finalStandardUrl = finalStandardUrl.uri;
      } else if (typeof finalStandardUrl === 'number') {
        const asset = await Asset.loadAsync(finalStandardUrl);
        finalStandardUrl = asset[0].localUri || asset[0].uri;
      }
      if (Platform.OS === 'web' && typeof finalStandardUrl === 'string' && finalStandardUrl.startsWith('/')) {
          finalStandardUrl = window.location.origin + finalStandardUrl;
      }

      console.log(`[CHECKPOINT ${index}] Standard URL configured.`);

      formData.append('standard_image_url', String(finalStandardUrl));
      formData.append('engine', selectedEngine);

      if (Platform.OS === 'web') {
        const res = await fetch(actualImage);
        const blob = await res.blob();
        formData.append('actual_image', blob, `actual_scan_${index}.jpg`);
      } else {
        formData.append('actual_image', { uri: actualImage, name: `actual_scan_${index}.jpg`, type: 'image/jpeg' });
      }

      console.log(`[CHECKPOINT ${index}] Sending POST request to Render...`);

      // 🚨 THE FIX: No 'headers' block used here at all! 
      const response = await fetch('https://api-g4yu2qwvwa-uc.a.run.app/api/audit-zone', { 
          method: 'POST', 
          body: formData 
      });

      console.log(`[CHECKPOINT ${index}] Server responded with status:`, response.status);

      if (!response.ok) { 
          const errText = await response.text(); 
          throw new Error(`Server Error ${response.status}: ${errText}`); 
      }

      const data = await response.json();
      setAiResult(data);

      if (data && data.score !== undefined) {
        onResult({ score: data.score, feedback: data.feedback, analysis_type: data.analysis_type, photoUri: actualImage });
      }
    } catch (error) {
      console.error(`[CHECKPOINT ${index}] CATCH ERROR:`, error);
      onError(`Failed to connect to AI engine.\n\n${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const standardImageSource = typeof point.standardImg === 'string' ? { uri: point.standardImg } : point.standardImg;

  // Render markdown bold text recursively
  const renderFormattedText = (text) => {
    if (!text) return null;
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <Text key={i} style={{ fontWeight: '900', color: COLORS.charcoal }}>{part.slice(2, -2)}</Text>;
      }
      return <Text key={i}>{part}</Text>;
    });
  };

  return (
    <View style={[styles.formSection, isLast && { borderBottomWidth: 0, paddingBottom: 0 }]}>
      <View style={styles.sectionHeader}>
        <View style={styles.avatarCircle}>
            <Text style={styles.avatarText}>{index}</Text>
        </View>
        <View style={{flex: 1}}>
            <Text style={styles.cardTitle}>{point.title}</Text>
            <Text style={styles.cardSub}>Checkpoint {index}</Text>
        </View>
        {aiResult && (
            <MaterialIcons name="check-circle" size={28} color={COLORS.primary} />
        )}
      </View>

      <View style={styles.comparisonRow}>
        <View style={styles.imageCol}>
          <Text style={styles.imageOverline}>STANDARD REQUIREMENT</Text>
          <View style={styles.imageFrame}><Image source={standardImageSource} style={styles.compareImage} /></View>
        </View>
        <View style={styles.imageCol}>
          <Text style={styles.imageOverline}>ACTUAL SCAN</Text>
          {actualImage ? (
            <TouchableOpacity onPress={takePicture} activeOpacity={0.8} style={styles.retakeWrapper}>
              <View style={styles.imageFrame}><Image source={{ uri: actualImage }} style={styles.compareImage} /></View>
              <View style={styles.retakeOverlay}>
                  <MaterialIcons name="flip-camera-ios" size={16} color={COLORS.surface} />
                  <Text style={styles.retakeText}>Retake</Text>
              </View>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={styles.uploadSurface} onPress={takePicture} activeOpacity={0.7}>
              <View style={styles.uploadFab}><MaterialIcons name="add-a-photo" size={28} color={COLORS.primary} /></View>
              <Text style={styles.uploadTitle}>Capture Area</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

      <View style={styles.actionRow}>
        {actualImage && !aiResult && !isAnalyzing && (
            <TouchableOpacity style={styles.evaluateBtn} onPress={analyzeImages} activeOpacity={0.8}>
              <MaterialIcons name="auto-awesome" size={20} color={COLORS.surface} style={{ marginRight: 8 }} />
              <Text style={styles.evaluateBtnText}>Evaluate Checkpoint</Text>
            </TouchableOpacity>
        )}
        
        {isAnalyzing && (
          <View style={styles.analyzingSurface}>
              <ActivityIndicator size="small" color={COLORS.primary} />
              <Text style={styles.analyzingText}>AI is scanning area...</Text>
          </View>
        )}

        {aiResult && (
          <View style={[styles.resultSurface, { backgroundColor: aiResult.score >= 4 ? COLORS.successLight : aiResult.score >= 2.5 ? COLORS.warningLight : COLORS.errorLight }]}>
            <View style={styles.resultHeaderRow}>
              <View style={styles.scoreRow}>
                  <Text style={[styles.scoreResultValue, { color: aiResult.score >= 4 ? COLORS.success : aiResult.score >= 2.5 ? COLORS.warning : COLORS.error }]}>{aiResult.score}</Text>
                  <Text style={styles.scoreResultMax}>/ 5.0</Text>
              </View>
              {aiResult.analysis_type && (
                <View style={[styles.typeChip, { backgroundColor: COLORS.surface }]}>
                  <Text style={[styles.typeChipText, { color: aiResult.score >= 4 ? COLORS.success : aiResult.score >= 2.5 ? COLORS.warning : COLORS.error }]}>{aiResult.analysis_type.toUpperCase()}</Text>
                </View>
              )}
            </View>
            <View style={styles.feedbackContainer}>
                <Text style={styles.feedbackText}>{renderFormattedText(aiResult.feedback)}</Text>
            </View>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: COLORS.background },
  heroBackground: { position: 'absolute', top: 0, left: 0, right: 0, height: 320, backgroundColor: COLORS.primary, borderBottomLeftRadius: 40, borderBottomRightRadius: 40 },
  appBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'transparent', paddingHorizontal: 20, paddingTop: Platform.OS === 'android' ? 40 : 20, paddingBottom: 16, zIndex: 10 },
  appBarLeft: { flexDirection: 'row', alignItems: 'center' },
  iconButton: { padding: 8, borderRadius: 20, overflow: 'hidden', marginRight: 12 },
  iconButtonDark: { padding: 8, borderRadius: 20, overflow: 'hidden', backgroundColor: COLORS.background },
  appBarSubtitle: { fontSize: 11, fontWeight: '800', color: 'rgba(255,255,255,0.7)', letterSpacing: 2, marginBottom: 2 },
  appBarTitle: { fontSize: 24, fontWeight: '900', color: COLORS.surface },
  historyFab: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.15)', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 24, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)' },
  historyFabText: { color: COLORS.surface, fontSize: 14, fontWeight: '700', marginLeft: 8 },
  progressContainer: { paddingHorizontal: 24, paddingVertical: 10 },
  progressBarBg: { height: 6, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 3, overflow: 'hidden', marginBottom: 8 },
  progressBarFill: { height: '100%', backgroundColor: COLORS.surface, borderRadius: 3 },
  progressText: { fontSize: 13, fontWeight: '600', color: 'rgba(255,255,255,0.8)', textAlign: 'right' },
  scrollContent: { paddingHorizontal: 20, paddingTop: 10, paddingBottom: 120 },
  formSheet: { width: '100%', maxWidth: 900, alignSelf: 'center', backgroundColor: COLORS.surface, borderRadius: 24, padding: Platform.OS === 'web' && window.innerWidth > 600 ? 40 : 25, elevation: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.1, shadowRadius: 24, marginBottom: 40 },
  formSection: { paddingVertical: 30, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  sectionTitle: { fontSize: 20, fontWeight: '900', color: COLORS.textHigh, marginBottom: 20 },
  materialInputContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.background, borderTopLeftRadius: 8, borderTopRightRadius: 8, borderBottomWidth: 2, borderBottomColor: COLORS.primary, paddingHorizontal: 16, paddingVertical: 10, marginBottom: 16 },
  inputIcon: { marginRight: 12 },
  inputWrapper: { flex: 1 },
  inputLabel: { fontSize: 11, color: COLORS.textMedium, fontWeight: '700', marginBottom: 2, textTransform: 'uppercase' },
  materialInput: { fontSize: 16, color: COLORS.textHigh, fontWeight: '600', outlineStyle: 'none', padding: 0 },
  metaChipsRow: { flexDirection: 'row', gap: 12 },
  metaChip: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#F0FDFA', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16 },
  metaChipText: { fontSize: 13, color: COLORS.primary, fontWeight: '700', marginLeft: 6 },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 25 },
  avatarCircle: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#F0FDFA', justifyContent: 'center', alignItems: 'center', marginRight: 16 },
  avatarText: { color: COLORS.primary, fontWeight: '900', fontSize: 18 },
  cardTitle: { fontSize: 20, fontWeight: '900', color: COLORS.textHigh },
  cardSub: { fontSize: 13, color: COLORS.textMedium, fontWeight: '600', marginTop: 2 },
  comparisonRow: { flexDirection: Platform.OS === 'web' && window.innerWidth > 600 ? 'row' : 'column', gap: 24, marginBottom: 24 },
  imageCol: { flex: 1 },
  imageOverline: { fontSize: 11, fontWeight: '800', color: COLORS.textMedium, marginBottom: 10, letterSpacing: 1 },
  imageFrame: { backgroundColor: COLORS.background, borderRadius: 16, overflow: 'hidden', borderWidth: 1, borderColor: COLORS.border },
  compareImage: { width: '100%', height: 220, resizeMode: 'cover' },
  uploadSurface: { width: '100%', height: 220, borderRadius: 16, backgroundColor: '#F8FAFC', borderWidth: 2, borderColor: COLORS.border, borderStyle: 'dashed', justifyContent: 'center', alignItems: 'center' },
  uploadFab: { width: 60, height: 60, borderRadius: 30, backgroundColor: '#F0FDFA', justifyContent: 'center', alignItems: 'center', marginBottom: 16 },
  uploadTitle: { color: COLORS.textHigh, fontSize: 16, fontWeight: '800' },
  retakeWrapper: { position: 'relative' },
  retakeOverlay: { position: 'absolute', bottom: 12, right: 12, backgroundColor: 'rgba(15, 23, 42, 0.75)', paddingHorizontal: 14, paddingVertical: 10, borderRadius: 20, flexDirection: 'row', alignItems: 'center' },
  retakeText: { color: COLORS.surface, fontSize: 13, fontWeight: '800', marginLeft: 6 },
  actionRow: { marginTop: 10 },
  evaluateBtn: { flexDirection: 'row', backgroundColor: COLORS.textHigh, paddingVertical: 18, borderRadius: 16, justifyContent: 'center', alignItems: 'center', elevation: 4, shadowColor: COLORS.textHigh, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.2, shadowRadius: 8 },
  evaluateBtnText: { color: COLORS.surface, fontWeight: '800', fontSize: 16, letterSpacing: 0.5 },
  analyzingSurface: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', backgroundColor: '#F0FDFA', paddingVertical: 18, borderRadius: 16, borderWidth: 1, borderColor: '#CCFBF1' },
  analyzingText: { marginLeft: 12, color: COLORS.primary, fontWeight: '800', fontSize: 16 },
  resultSurface: { padding: 24, borderRadius: 16 },
  resultHeaderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 15 },
  scoreRow: { flexDirection: 'row', alignItems: 'baseline' },
  scoreResultValue: { fontWeight: '900', fontSize: 36, letterSpacing: -1 },
  scoreResultMax: { color: COLORS.textMedium, fontWeight: '800', fontSize: 18, marginLeft: 4 },
  typeChip: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 12, elevation: 1 },
  typeChipText: { fontSize: 12, fontWeight: '900', letterSpacing: 0.5 },
  feedbackContainer: { marginTop: 8, paddingHorizontal: 4 },
  feedbackText: { color: COLORS.textHigh, fontSize: 14, lineHeight: 24, fontWeight: '500' },
  bottomAppBar: { position: 'absolute', bottom: 0, left: 0, right: 0, backgroundColor: COLORS.surface, paddingHorizontal: 24, paddingVertical: 16, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', elevation: 16, shadowColor: '#000', shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.1, shadowRadius: 12 },
  scoreDisplay: { flexDirection: 'column' },
  scoreDisplayLabel: { fontSize: 11, color: COLORS.textMedium, fontWeight: '800', textTransform: 'uppercase', letterSpacing: 0.5 },
  scoreDisplayValue: { fontSize: 26, fontWeight: '900', color: COLORS.primary },
  fabExtended: { flexDirection: 'row', backgroundColor: COLORS.primary, paddingHorizontal: 24, paddingVertical: 14, borderRadius: 16, alignItems: 'center', elevation: 6, shadowColor: COLORS.primary, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 8 },
  fabDisabled: { backgroundColor: COLORS.border, elevation: 0, shadowOpacity: 0 },
  fabExtendedText: { color: COLORS.surface, fontWeight: '900', fontSize: 16, letterSpacing: 0.5 },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(15, 23, 42, 0.7)', justifyContent: 'center', alignItems: 'center', padding: 24 },
  dialogCard: { backgroundColor: COLORS.surface, width: '100%', maxWidth: 380, borderRadius: 28, padding: 32, alignItems: 'center', elevation: 24, shadowColor: '#000', shadowOffset: { width: 0, height: 12 }, shadowOpacity: 0.3, shadowRadius: 24 },
  dialogIconBox: { width: 72, height: 72, borderRadius: 36, justifyContent: 'center', alignItems: 'center', marginBottom: 20 },
  dialogTitle: { fontSize: 22, fontWeight: '900', color: COLORS.textHigh, marginBottom: 12, textAlign: 'center' },
  dialogMessage: { fontSize: 16, color: COLORS.textMedium, textAlign: 'center', lineHeight: 24, marginBottom: 32 },
  dialogButton: { backgroundColor: COLORS.textHigh, width: '100%', paddingVertical: 16, borderRadius: 16, alignItems: 'center' },
  dialogButtonText: { color: COLORS.surface, fontWeight: '900', fontSize: 16, letterSpacing: 0.5 },
  vaultModalOverlay: { flex: 1, backgroundColor: 'rgba(15, 23, 42, 0.8)', justifyContent: 'flex-end', alignItems: 'center' },
  vaultModalCard: { backgroundColor: COLORS.surface, width: '100%', maxWidth: 600, height: '85%', borderTopLeftRadius: 32, borderTopRightRadius: 32, elevation: 24 },
  vaultModalHeader: { flexDirection: 'row', alignItems: 'center', padding: 24, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  vaultModalBadge: { fontSize: 11, fontWeight: '900', color: COLORS.primary, letterSpacing: 1.5, marginBottom: 4 },
  vaultModalTitle: { fontSize: 22, fontWeight: '900', color: COLORS.textHigh },
  vaultModalScroll: { padding: 24 },
  timelineRow: { flexDirection: 'row', marginBottom: 0 },
  timelineDotBox: { width: 32, alignItems: 'center' },
  timelineDot: { width: 16, height: 16, borderRadius: 8, zIndex: 2 },
  timelineLine: { width: 2, flex: 1, backgroundColor: COLORS.border, marginTop: -4, marginBottom: -4 },
  timelineContent: { flex: 1, paddingBottom: 32, paddingLeft: 12 },
  timelineTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, marginTop: -4 },
  timelineDate: { fontSize: 16, fontWeight: '800', color: COLORS.textHigh },
  timelineScoreBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  timelineScoreText: { fontSize: 13, fontWeight: '900' },
  timelineAuditor: { fontSize: 14, color: COLORS.textMedium, fontWeight: '600' },
});