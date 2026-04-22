import React, { useState } from 'react';
import { StyleSheet, Text, View, SafeAreaView, TouchableOpacity, ScrollView, Platform, StatusBar, TextInput, Image, Modal } from 'react-native';
import { Feather } from '@expo/vector-icons';

const COLORS = {
  charcoal: '#0F172A', teal: '#0D9488', pink: '#E11D48', 
  bg: '#F8FAFC', white: '#FFFFFF', border: '#E2E8F0',
  textMuted: '#64748B', orange: '#F59E0B', green: '#10B981', red: '#EF4444'
};

export default function ZonesOverview({ userEmail, zones, auditResults, auditHistory, onSelectZone, onNavigate, onLogout }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [zoneVaultModal, setZoneVaultModal] = useState(null); // Tracks which zone's history to show
  const isMobile = Platform.OS === 'web' ? window.innerWidth <= 600 : false;

  // Filter zones by search
  const filteredZones = zones.filter(zone => 
    zone.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    zone.badge.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Get history for a specific zone
  const getZoneHistory = (zoneId) => {
    if (!auditHistory) return [];
    return auditHistory.filter(record => record.zoneId === zoneId);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.white} />
      
      {/* GLOBAL TOP NAV */}
      <View style={styles.topNav}>
        <View style={styles.navLeft}>
            <View style={styles.logoBox}><Image source={require('./assets/logo.png')} style={styles.logoImage} /></View>
        </View>

        <View style={styles.navCenter}>
            <TouchableOpacity style={styles.navMenuBtn} onPress={() => onNavigate('dashboard')}>
                <Feather name="pie-chart" size={16} color={COLORS.textMuted} />
                {!isMobile && <Text style={styles.navMenuText}>Dashboard</Text>}
            </TouchableOpacity>

            <TouchableOpacity style={styles.navMenuBtn} onPress={() => onNavigate('leaderboard')}>
                <Feather name="award" size={16} color={COLORS.textMuted} />
                {!isMobile && <Text style={styles.navMenuText}>Leaderboard</Text>}
            </TouchableOpacity>
            
            <TouchableOpacity style={[styles.navMenuBtn, styles.navMenuBtnActive]} disabled>
                <Feather name="grid" size={16} color={COLORS.teal} />
                {!isMobile && <Text style={[styles.navMenuText, styles.navMenuTextActive]}>Zones</Text>}
            </TouchableOpacity>
        </View>

        <View style={styles.navRight}>
            <TouchableOpacity style={styles.navActionBtn} onPress={() => onNavigate('history')}>
                <Feather name="archive" size={14} color={COLORS.textMuted} />
                {!isMobile && <Text style={styles.navActionText}>Global Vault</Text>}
            </TouchableOpacity>
            <TouchableOpacity style={styles.explicitLogoutBtn} onPress={onLogout}>
                <Feather name="log-out" size={14} color={COLORS.pink} />
            </TouchableOpacity>
        </View>
      </View>

      <View style={styles.mainContainer}>
        {/* HEADER & SEARCH */}
        <View style={styles.headerRow}>
            <View>
                <Text style={styles.welcomeText}>Zones Overview</Text>
                <Text style={styles.dateText}>Select a factory zone to begin auditing or view its history</Text>
            </View>
            <View style={styles.searchBox}>
                <Feather name="search" size={18} color={COLORS.textMuted} />
                <TextInput 
                    style={styles.searchInput} 
                    placeholder="Search zones..." 
                    value={searchQuery}
                    onChangeText={setSearchQuery}
                    placeholderTextColor={COLORS.textMuted}
                />
            </View>
        </View>

        {/* ZONES GRID */}
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          <View style={styles.grid}>
            {filteredZones.map((zone) => {
              const result = auditResults ? auditResults[zone.id] : null;
              const hasCompleted = !!result;
              const score = result ? parseFloat(result.score) : 0;
              const isExcellent = score >= 4.0;
              const isWarning = score >= 2.5 && score < 4.0;
              
              const zonePastRecords = getZoneHistory(zone.id);
              const hasHistory = zonePastRecords.length > 0;

              return (
                <View key={zone.id} style={styles.zoneCard}>
                  {/* Card Image Header */}
                  <View style={styles.cardImageContainer}>
                    <Image source={typeof zone.image === 'string' ? { uri: zone.image } : zone.image} style={styles.cardImage} />
                    <View style={styles.cardOverlay} />
                    
                    <View style={styles.statusBadgeTop}>
                        <View style={[styles.statusDot, { backgroundColor: hasCompleted ? (isExcellent ? COLORS.green : isWarning ? COLORS.orange : COLORS.red) : COLORS.textMuted }]} />
                        <Text style={styles.statusBadgeTextTop}>
                            {hasCompleted ? (isExcellent ? 'EXCELLENT' : isWarning ? 'NEEDS WORK' : 'CRITICAL') : 'PENDING AUDIT'}
                        </Text>
                    </View>

                    <View style={styles.badgeBottomRow}>
                        <Text style={styles.cardBadge}>{zone.badge}</Text>
                    </View>
                  </View>

                  {/* Card Body */}
                  <View style={styles.cardBody}>
                    <View style={styles.cardBodyTop}>
                        <Text style={styles.zoneNameText}>{zone.name}</Text>
                        <View style={[styles.scoreCircle, hasCompleted && { borderColor: isExcellent ? COLORS.green : isWarning ? COLORS.orange : COLORS.red }]}>
                            <Text style={[styles.scoreValue, hasCompleted && { color: isExcellent ? COLORS.green : isWarning ? COLORS.orange : COLORS.red }]}>
                                {hasCompleted ? score.toFixed(1) : '--'}
                            </Text>
                        </View>
                    </View>

                    <View style={styles.cardFooter}>
                        <View>
                            <Text style={styles.lastAuditLabel}>LAST AUDIT</Text>
                            <Text style={styles.lastAuditDate}>{hasCompleted ? result.date : 'Not Started'}</Text>
                        </View>
                        
                        <View style={styles.actionButtonsRow}>
                            {/* --- NEW: ZONE VAULT BUTTON --- */}
                            {hasHistory && (
                                <TouchableOpacity 
                                    style={styles.zoneVaultBtn} 
                                    onPress={() => setZoneVaultModal({ zone, records: zonePastRecords })}
                                    title="View Zone History"
                                >
                                    <Feather name="archive" size={16} color={COLORS.textMuted} />
                                </TouchableOpacity>
                            )}

                            <TouchableOpacity 
                                style={[styles.startAuditBtn, hasCompleted && styles.reAuditBtn]} 
                                onPress={() => onSelectZone(zone.id)}
                            >
                                <Text style={[styles.startAuditBtnText, hasCompleted && styles.reAuditBtnText]}>
                                    {hasCompleted ? 'Re-Audit' : 'Start Audit'}
                                </Text>
                                <Feather name="chevron-right" size={16} color={hasCompleted ? COLORS.teal : COLORS.white} />
                            </TouchableOpacity>
                        </View>
                    </View>
                  </View>
                </View>
              );
            })}
          </View>
        </ScrollView>
      </View>

      {/* --- NEW: ZONE-SPECIFIC VAULT MODAL --- */}
      <Modal transparent={true} visible={!!zoneVaultModal} animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            
            {/* Modal Header */}
            <View style={styles.modalHeader}>
                <View style={{flex: 1}}>
                    <Text style={styles.modalBadge}>{zoneVaultModal?.zone.badge} VAULT</Text>
                    <Text style={styles.modalTitle}>{zoneVaultModal?.zone.name} History</Text>
                </View>
                <TouchableOpacity onPress={() => setZoneVaultModal(null)} style={styles.closeBtn}>
                    <Feather name="x" size={24} color={COLORS.charcoal} />
                </TouchableOpacity>
            </View>

            {/* Timeline Content */}
            <ScrollView style={styles.modalScroll} showsVerticalScrollIndicator={false}>
                {zoneVaultModal?.records.map((record, index) => (
                    <View key={record.id} style={styles.timelineRow}>
                        <View style={styles.timelineDotBox}>
                            <View style={[styles.timelineDot, { backgroundColor: record.score >= 4 ? COLORS.green : record.score >= 2.5 ? COLORS.orange : COLORS.red }]} />
                            {index !== zoneVaultModal.records.length - 1 && <View style={styles.timelineLine} />}
                        </View>
                        <View style={styles.timelineContent}>
                            <View style={styles.timelineTop}>
                                <Text style={styles.timelineDate}>{record.date}</Text>
                                <View style={[styles.timelineScoreBadge, { backgroundColor: record.score >= 4 ? '#ECFDF5' : record.score >= 2.5 ? '#FFFBEB' : '#FEF2F2' }]}>
                                    <Text style={[styles.timelineScoreText, { color: record.score >= 4 ? COLORS.green : record.score >= 2.5 ? COLORS.orange : COLORS.red }]}>{record.score} / 5.0</Text>
                                </View>
                            </View>
                            <Text style={styles.timelineAuditor}>Auditor: {record.auditor}</Text>
                        </View>
                    </View>
                ))}
            </ScrollView>
            
            <View style={styles.modalFooter}>
                <Text style={styles.modalFooterText}>To download full PDF reports with images, please visit the Global Vault.</Text>
            </View>

          </View>
        </View>
      </Modal>

    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: COLORS.bg },
  topNav: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: COLORS.white, paddingHorizontal: 20, height: Platform.OS === 'android' ? 120 : 100, paddingTop: Platform.OS === 'android' ? 25 : 0, borderBottomWidth: 1, borderBottomColor: COLORS.border, zIndex: 10 },
  navLeft: { flexDirection: 'row', alignItems: 'center', width: Platform.OS === 'web' && window.innerWidth > 800 ? 250 : 'auto' },
  logoBox: { width: 160, height: 60, justifyContent: 'center', alignItems: 'flex-start' },
  logoImage: { width: '100%', height: '100%', resizeMode: 'contain' },
  navCenter: { flexDirection: 'row', gap: 10, flex: 1, justifyContent: 'center' },
  navMenuBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 12 },
  navMenuBtnActive: { backgroundColor: '#F0FDFA' },
  navMenuText: { color: COLORS.textMuted, fontWeight: '700', fontSize: 14, marginLeft: 8 },
  navMenuTextActive: { color: COLORS.teal },
  navRight: { flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-end', gap: 12, width: Platform.OS === 'web' && window.innerWidth > 800 ? 250 : 'auto' },
  navActionBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, backgroundColor: '#F8FAFC' },
  navActionText: { color: COLORS.textMuted, fontWeight: '700', fontSize: 13, marginLeft: 6 },
  explicitLogoutBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, backgroundColor: '#FFF1F2' },
  
  mainContainer: { flex: 1, maxWidth: 1200, width: '100%', alignSelf: 'center', paddingTop: 40, paddingHorizontal: 20 },
  headerRow: { flexDirection: Platform.OS === 'web' && window.innerWidth > 800 ? 'row' : 'column', justifyContent: 'space-between', alignItems: Platform.OS === 'web' && window.innerWidth > 800 ? 'center' : 'flex-start', marginBottom: 30, gap: 15 },
  welcomeText: { fontSize: 32, fontWeight: '900', color: COLORS.charcoal, letterSpacing: -0.5, marginBottom: 8 },
  dateText: { fontSize: 16, color: COLORS.textMuted, fontWeight: '600' },
  searchBox: { flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.white, paddingHorizontal: 16, paddingVertical: 12, borderRadius: 12, borderWidth: 1, borderColor: COLORS.border, width: Platform.OS === 'web' && window.innerWidth > 800 ? 350 : '100%', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.02, shadowRadius: 6 },
  searchInput: { flex: 1, marginLeft: 10, fontSize: 15, color: COLORS.charcoal, outlineStyle: 'none' },
  
  scrollContent: { paddingBottom: 60 },
  grid: { 
    flexDirection: 'row', 
    flexWrap: 'wrap', 
    gap: 20, 
    justifyContent: 'flex-start' 
  },
  
  zoneCard: { 
    backgroundColor: COLORS.white, 
    width: Platform.OS === 'web' && window.innerWidth > 1000 ? 'calc(25% - 15px)' : Platform.OS === 'web' && window.innerWidth > 700 ? 'calc(50% - 10px)' : '100%', 
    borderRadius: 20, 
    overflow: 'hidden', 
    borderWidth: 1, 
    borderColor: COLORS.border, 
    shadowColor: '#000', 
    shadowOffset: { width: 0, height: 4 }, 
    shadowOpacity: 0.04, 
    shadowRadius: 10, 
    elevation: 3, 
    marginBottom: 10 
  },
  cardImageContainer: { width: '100%', height: 160, position: 'relative' },
  cardImage: { width: '100%', height: '100%', resizeMode: 'cover' },
  cardOverlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(15, 23, 42, 0.3)' },
  statusBadgeTop: { position: 'absolute', top: 15, left: 15, backgroundColor: COLORS.white, paddingHorizontal: 10, paddingVertical: 6, borderRadius: 20, flexDirection: 'row', alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4 },
  statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  statusBadgeTextTop: { fontSize: 10, fontWeight: '900', color: COLORS.charcoal, letterSpacing: 0.5 },
  badgeBottomRow: { position: 'absolute', bottom: 15, left: 15, backgroundColor: COLORS.white, paddingHorizontal: 10, paddingVertical: 4, borderRadius: 6 },
  cardBadge: { fontSize: 11, fontWeight: '900', color: COLORS.charcoal, letterSpacing: 0.5 },
  
  cardBody: { padding: 20 },
  cardBodyTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
  zoneNameText: { fontSize: 20, fontWeight: '900', color: COLORS.charcoal },
  scoreCircle: { width: 44, height: 44, borderRadius: 22, borderWidth: 2, borderColor: '#E2E8F0', justifyContent: 'center', alignItems: 'center' },
  scoreValue: { fontSize: 15, fontWeight: '900', color: '#94A3B8' },
  
  cardFooter: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end' },
  lastAuditLabel: { fontSize: 10, fontWeight: '800', color: COLORS.textMuted, letterSpacing: 1, marginBottom: 4 },
  lastAuditDate: { fontSize: 13, fontWeight: '600', color: COLORS.charcoal },
  
  actionButtonsRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  zoneVaultBtn: { padding: 10, borderRadius: 10, borderWidth: 1, borderColor: COLORS.border, backgroundColor: '#F8FAFC' },
  startAuditBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.charcoal, paddingHorizontal: 16, paddingVertical: 10, borderRadius: 10 },
  startAuditBtnText: { color: COLORS.white, fontSize: 13, fontWeight: '800', marginRight: 4 },
  reAuditBtn: { backgroundColor: '#F0FDFA', borderWidth: 1, borderColor: '#CCFBF1' },
  reAuditBtnText: { color: COLORS.teal },

  // --- MODAL STYLES ---
  modalOverlay: { flex: 1, backgroundColor: 'rgba(15, 23, 42, 0.7)', justifyContent: 'center', alignItems: 'center', padding: 20 },
  modalCard: { backgroundColor: COLORS.white, width: '100%', maxWidth: 500, maxHeight: '80%', borderRadius: 24, overflow: 'hidden', shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.3, shadowRadius: 30, elevation: 20 },
  modalHeader: { flexDirection: 'row', alignItems: 'flex-start', padding: 25, backgroundColor: '#F8FAFC', borderBottomWidth: 1, borderBottomColor: COLORS.border },
  modalBadge: { fontSize: 12, fontWeight: '800', color: COLORS.teal, letterSpacing: 1, marginBottom: 6 },
  modalTitle: { fontSize: 24, fontWeight: '900', color: COLORS.charcoal },
  closeBtn: { padding: 8, backgroundColor: '#E2E8F0', borderRadius: 20 },
  
  modalScroll: { padding: 25 },
  timelineRow: { flexDirection: 'row', marginBottom: 0 },
  timelineDotBox: { width: 30, alignItems: 'center' },
  timelineDot: { width: 14, height: 14, borderRadius: 7, zIndex: 2 },
  timelineLine: { width: 2, flex: 1, backgroundColor: '#E2E8F0', marginTop: -2, marginBottom: -2 },
  timelineContent: { flex: 1, paddingBottom: 30, paddingLeft: 10 },
  timelineTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, marginTop: -4 },
  timelineDate: { fontSize: 16, fontWeight: '800', color: COLORS.charcoal },
  timelineScoreBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  timelineScoreText: { fontSize: 13, fontWeight: '900' },
  timelineAuditor: { fontSize: 14, color: COLORS.textMuted, fontWeight: '500' },
  
  modalFooter: { padding: 20, backgroundColor: '#F8FAFC', borderTopWidth: 1, borderTopColor: COLORS.border, alignItems: 'center' },
  modalFooterText: { fontSize: 12, color: COLORS.textMuted, textAlign: 'center', fontStyle: 'italic' }
});