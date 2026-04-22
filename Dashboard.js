import React, { useState } from 'react';
import { StyleSheet, Text, View, SafeAreaView, Platform, StatusBar, TouchableOpacity, useWindowDimensions, Modal, Image, ScrollView } from 'react-native';
import { Feather } from '@expo/vector-icons';
import { Svg, Polygon, Line, Text as SvgText, Circle } from 'react-native-svg';

const COLORS = {
  charcoal: '#0F172A', teal: '#0D9488', pink: '#E11D48',
  bg: '#F8FAFC', white: '#FFFFFF', border: '#E2E8F0',
  textMuted: '#64748B', orange: '#F59E0B', green: '#10B981', red: '#EF4444'
};

export default function Dashboard({ userEmail, zones, auditResults, auditHistory, onNavigate, onLogout, onArchiveWeek }) {
  const [modalVisible, setModalVisible] = useState(false);
  const [timeFilter, setTimeFilter] = useState('current');
  const { width } = useWindowDimensions();
  const isMobile = width <= 800; // Adjusted breakpoint slightly for the new menu

  const safeEmail = userEmail || 'Auditor';
  const formattedName = safeEmail.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

  // --- DYNAMIC GREETING ---
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening';
  const greetingEmoji = hour < 12 ? '☀️' : hour < 18 ? '🌤️' : '🌙';

  const today = new Date();
  const dateOptions = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
  const formattedFullDate = today.toLocaleDateString('en-US', dateOptions);

  // --- TIMEFRAME FILTERING LOGIC ---
  let activeData = {};

  if (timeFilter === 'current') {
    activeData = auditResults || {};
  } else {
    const now = Date.now();
    const cutoff = timeFilter === '7days' ? (7 * 24 * 60 * 60 * 1000) : (30 * 24 * 60 * 60 * 1000);

    (auditHistory || []).forEach(record => {
      const recordTime = parseInt(record.id);
      if (now - recordTime <= cutoff) {
        if (!activeData[record.zoneId] || parseInt(activeData[record.zoneId].id) < recordTime) {
          activeData[record.zoneId] = { score: record.score };
        }
      }
    });
  }

  // --- CALCULATE SUMMARY METRICS ---
  const completedZonesCount = Object.keys(activeData).length;
  let overallCompliance = 0;
  let criticalZonesCount = 0;

  if (completedZonesCount > 0) {
    const scores = Object.values(activeData).map(r => parseFloat(r.score));
    const sumScores = scores.reduce((a, b) => a + b, 0);
    overallCompliance = Math.round((sumScores / completedZonesCount / 5.0) * 100);
    criticalZonesCount = scores.filter(s => s < 3.0).length;
  }

  // --- CUSTOM 35-POINT RADAR CHART MATH ---
  const chartSize = isMobile ? width - 10 : Math.min(width - 60, 700);
  const center = chartSize / 2;
  const maxRadius = center - 115; 
  const numAxes = zones?.length || 35;

  const getCoordinates = (scoreValue, index) => {
    const angle = (Math.PI * 2 * index) / numAxes - Math.PI / 2; 
    const distance = (scoreValue / 5.0) * maxRadius;
    return {
      x: center + distance * Math.cos(angle),
      y: center + distance * Math.sin(angle)
    };
  };

  const dataPoints = (zones || []).map((zone, index) => {
    const scoreData = activeData[zone.id];
    const score = scoreData ? parseFloat(scoreData.score) : 0;
    return getCoordinates(score, index);
  });
  const dataPolygonString = dataPoints.map(p => `${p.x},${p.y}`).join(' ');

  // --- ROBUST LABEL COLLISION RESOLUTION ---
  const initialLabels = (zones || []).map((zone, index) => {
    const angle = (Math.PI * 2 * index) / numAxes - Math.PI / 2;
    let textAnchor = "middle";
    if (Math.cos(angle) > 0.1) textAnchor = "start";
    else if (Math.cos(angle) < -0.1) textAnchor = "end";

    const baseRadius = maxRadius + 18;
    let lx = center + baseRadius * Math.cos(angle);
    let ly = center + baseRadius * Math.sin(angle);

    const isTopOrBottom = Math.abs(Math.cos(angle)) < 0.3; 
    if (isTopOrBottom) {
      ly += Math.sign(Math.sin(angle)) * 18; 
      lx += Math.sign(Math.cos(angle)) * 25;
      textAnchor = Math.cos(angle) > 0 ? "start" : Math.cos(angle) < 0 ? "end" : "middle";
    }

    const prettyBadgeName = zone.badge.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ');

    return {
      id: index, text: prettyBadgeName, angle, cx: center + maxRadius * Math.cos(angle), cy: center + maxRadius * Math.sin(angle),
      x: lx, y: ly, textAnchor, isRight: Math.cos(angle) >= 0
    };
  });

  const MIN_Y_GAP = 18; 
  const resolveOverlaps = (labelsSubset) => {
    labelsSubset.sort((a, b) => a.y - b.y);
    for (let iter = 0; iter < 15; iter++) { 
      for (let i = 0; i < labelsSubset.length - 1; i++) {
        const diff = labelsSubset[i + 1].y - labelsSubset[i].y;
        if (diff < MIN_Y_GAP) {
          const shift = (MIN_Y_GAP - diff) / 2;
          labelsSubset[i].y -= shift;
          labelsSubset[i + 1].y += shift;
        }
      }
    }
  };

  const rightLabels = initialLabels.filter(l => l.isRight);
  const leftLabels = initialLabels.filter(l => !l.isRight);
  resolveOverlaps(rightLabels);
  resolveOverlaps(leftLabels);

  const finalLabels = [...rightLabels, ...leftLabels].sort((a, b) => a.id - b.id);

  const handleArchiveConfirm = () => {
    setModalVisible(false);
    if (onArchiveWeek) onArchiveWeek();
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.white} />

      {/* TOP NAV & GREETING */}
      <View style={styles.topNav}>
        <View style={styles.navLeft}>
          <View style={styles.logoBox}><Image source={require('./assets/logo.png')} style={styles.logoImage} /></View>
          {!isMobile && (
            <View style={styles.greetingBox}>
              <Text style={styles.greetingEmoji}>{greetingEmoji}</Text>
              <Text style={styles.greetingTime}>{greeting},</Text>
              <Text style={styles.greetingName}>{formattedName}</Text>
            </View>
          )}
        </View>

        {/* --- NEW: CENTRAL NAVIGATION MENU --- */}
        <View style={styles.navCenter}>
            <TouchableOpacity style={[styles.navMenuBtn, styles.navMenuBtnActive]} disabled>
                <Feather name="pie-chart" size={16} color={COLORS.teal} />
                {!isMobile && <Text style={[styles.navMenuText, styles.navMenuTextActive]}>Dashboard</Text>}
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.navMenuBtn} onPress={() => onNavigate('leaderboard')}>
                <Feather name="award" size={16} color={COLORS.textMuted} />
                {!isMobile && <Text style={styles.navMenuText}>Leaderboard</Text>}
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.navMenuBtn} onPress={() => onNavigate('zones')}>
                <Feather name="grid" size={16} color={COLORS.textMuted} />
                {!isMobile && <Text style={styles.navMenuText}>Zones</Text>}
            </TouchableOpacity>
        </View>

        <View style={styles.navRight}>
          <TouchableOpacity style={styles.navActionBtn} onPress={() => onNavigate('history')}>
            <Feather name="archive" size={14} color={COLORS.textMuted} />
            {!isMobile && <Text style={styles.navActionText}>Vault</Text>}
          </TouchableOpacity>
          <TouchableOpacity style={styles.explicitLogoutBtn} onPress={onLogout}>
            <Feather name="log-out" size={14} color={COLORS.pink} />
            {!isMobile && <Text style={styles.explicitLogoutText}>Log out</Text>}
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView style={styles.mainContainer} showsVerticalScrollIndicator={false}>

        {/* PAGE HEADER */}
        <View style={styles.pageHeader}>
          <View>
            <Text style={styles.welcomeText}>Dashboard Overview</Text>
            <Text style={styles.dateText}>{formattedFullDate}</Text>
          </View>
          <TouchableOpacity style={styles.newWeekBtn} onPress={() => setModalVisible(true)}>
            <Feather name="refresh-cw" size={14} color={COLORS.charcoal} />
            <Text style={styles.newWeekText}>Start New Week</Text>
          </TouchableOpacity>
        </View>

        {/* CTA CARD */}
        <View style={styles.ctaCard}>
          <View style={styles.ctaTextCol}>
            <Text style={styles.ctaTitle}>Ready to begin your factory walk?</Text>
            <Text style={styles.ctaSub}>Access the full grid of all 35 zones, view live statuses, and run AI Vision Analysis on individual checklists.</Text>
          </View>
          <TouchableOpacity style={styles.ctaBtn} onPress={() => onNavigate('zones')} activeOpacity={0.8}>
            <Text style={styles.ctaBtnText}>Browse All Zones</Text>
            <Feather name="arrow-right" size={16} color={COLORS.white} style={{ marginLeft: 8 }} />
          </TouchableOpacity>
        </View>

        {/* TIMEFRAME TOGGLE */}
        <View style={styles.filterContainer}>
          <TouchableOpacity style={[styles.filterBtn, timeFilter === 'current' && styles.filterBtnActive]} onPress={() => setTimeFilter('current')}>
            <Text style={[styles.filterText, timeFilter === 'current' && styles.filterTextActive]}>Current Week</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.filterBtn, timeFilter === '7days' && styles.filterBtnActive]} onPress={() => setTimeFilter('7days')}>
            <Text style={[styles.filterText, timeFilter === '7days' && styles.filterTextActive]}>Past 7 Days</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.filterBtn, timeFilter === '30days' && styles.filterBtnActive]} onPress={() => setTimeFilter('30days')}>
            <Text style={[styles.filterText, timeFilter === '30days' && styles.filterTextActive]}>Past 30 Days</Text>
          </TouchableOpacity>
        </View>

        {/* METRICS ROW */}
        <View style={styles.metricsContainer}>
          <View style={styles.metricCard}>
            <Text style={styles.metricTitle}>Overall 5S Score</Text>
            <View style={styles.numberRow}>
              <Text style={[styles.metricBigNumber, { color: overallCompliance >= 80 ? COLORS.green : overallCompliance >= 50 ? COLORS.orange : COLORS.red }]}>
                {completedZonesCount > 0 ? `${overallCompliance}%` : 'N/A'}
              </Text>
            </View>
            <Text style={styles.metricSubText}>{completedZonesCount > 0 ? 'Based on selected timeframe' : 'No audits found'}</Text>
          </View>

          <View style={styles.metricCard}>
            <Text style={styles.metricTitle}>Audit Progress</Text>
            <View style={styles.numberRow}>
              <Text style={[styles.metricBigNumber, { color: COLORS.charcoal }]}>{completedZonesCount}</Text>
              <Text style={styles.metricMaxNumber}> / {zones?.length || 35}</Text>
            </View>
            <Text style={styles.metricSubText}>Zones completed</Text>
          </View>

          <View style={styles.metricCard}>
            <Text style={styles.metricTitle}>Critical Zones</Text>
            <View style={styles.numberRow}>
              <Text style={[styles.metricBigNumber, { color: criticalZonesCount > 0 ? COLORS.red : COLORS.textMuted }]}>
                {criticalZonesCount}
              </Text>
            </View>
            <Text style={styles.metricSubText}>Zones scoring below 3.0 / 5.0</Text>
          </View>
        </View>

        {/* CUSTOM RADAR CHART */}
        <View style={styles.chartCard}>
          <Text style={styles.chartTitle}>Full Factory Radar Overview</Text>
          <Text style={styles.chartSub}>Outer edge equals 5.0. Missing data pulls to center (0.0).</Text>

          <View style={{ alignItems: 'center', marginTop: 35, marginBottom: 25 }}>
            <Svg width={chartSize} height={chartSize} style={{ overflow: 'visible' }}>
              {[1, 2, 3, 4, 5].map(level => {
                const gridPoints = (zones || []).map((_, i) => {
                  const pos = getCoordinates(level, i);
                  return `${pos.x},${pos.y}`;
                }).join(' ');
                return (
                  <Polygon
                    key={`grid-${level}`} points={gridPoints}
                    stroke="#E2E8F0" strokeWidth="1"
                    fill={level % 2 === 0 ? "#F8FAFC" : "none"}
                  />
                );
              })}

              {(zones || []).map((_, i) => {
                const pos = getCoordinates(5, i);
                return <Line key={`axis-${i}`} x1={center} y1={center} x2={pos.x} y2={pos.y} stroke="#E2E8F0" strokeWidth="1" />;
              })}

              <Polygon
                points={dataPolygonString}
                fill="rgba(13, 148, 136, 0.25)"
                stroke={COLORS.teal}
                strokeWidth="2.5"
              />

              {dataPoints.map((pos, i) => (
                <Circle key={`point-${i}`} cx={pos.x} cy={pos.y} r="3.5" fill={COLORS.teal} />
              ))}

              {finalLabels.map((lbl, i) => {
                return (
                  <React.Fragment key={`label-group-${i}`}>
                    <Line
                      x1={lbl.cx} y1={lbl.cy}
                      x2={lbl.textAnchor === "start" ? lbl.x - 4 : lbl.textAnchor === "end" ? lbl.x + 4 : lbl.x}
                      y2={lbl.y - 3}
                      stroke="#E2E8F0" strokeWidth="1" strokeDasharray="2,2"
                    />
                    <SvgText
                      x={lbl.x} y={lbl.y + 1}
                      fontSize="10" fontWeight="700" fill={COLORS.textMuted} textAnchor={lbl.textAnchor}
                    >
                      {lbl.text}
                    </SvgText>
                  </React.Fragment>
                );
              })}
            </Svg>
          </View>
        </View>

        <View style={{ height: 60 }} />
      </ScrollView>

      {/* MODAL */}
      <Modal transparent={true} visible={modalVisible} animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <View style={[styles.modalIconBox, { backgroundColor: '#FFFBEB' }]}><Feather name="alert-triangle" size={32} color="#F59E0B" /></View>
            <Text style={styles.modalTitle}>Archive & Start New Week?</Text>
            <Text style={styles.modalMessage}>This will save your current progress to the History Vault and clear the board so you can start a fresh audit.</Text>
            <View style={styles.modalButtonGroup}>
              <TouchableOpacity style={styles.modalBtnCancel} onPress={() => setModalVisible(false)}><Text style={styles.modalBtnCancelText}>Cancel</Text></TouchableOpacity>
              <TouchableOpacity style={styles.modalBtnConfirm} onPress={handleArchiveConfirm}><Text style={styles.modalBtnConfirmText}>Archive Week</Text></TouchableOpacity>
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
  
  // Adjusted for standard web layout alignment
  navLeft: { flexDirection: 'row', alignItems: 'center', width: Platform.OS === 'web' && window.innerWidth > 800 ? 280 : 'auto' },
  logoBox: { width: 150, height: 50, justifyContent: 'center', alignItems: 'flex-start' }, 
  logoImage: { width: '100%', height: '100%', resizeMode: 'contain' },

  greetingBox: {
    marginLeft: 15,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#F1F5F9',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 100,
    borderWidth: 1,
    borderColor: '#E2E8F0'
  },
  greetingEmoji: { fontSize: 14, marginRight: 6 },
  greetingTime: { fontSize: 13, color: '#64748B', fontWeight: '600', marginRight: 4 },
  greetingName: { fontSize: 13, color: '#0F172A', fontWeight: '800', letterSpacing: -0.3 },

  // --- NEW: NAV CENTER STYLES ---
  navCenter: { flexDirection: 'row', gap: 10, flex: 1, justifyContent: 'center' },
  navMenuBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 12 },
  navMenuBtnActive: { backgroundColor: '#F0FDFA' },
  navMenuText: { color: COLORS.textMuted, fontWeight: '700', fontSize: 14, marginLeft: 8 },
  navMenuTextActive: { color: COLORS.teal },

  navRight: { flexDirection: 'row', alignItems: 'center', justifyContent: 'flex-end', gap: 12, width: Platform.OS === 'web' && window.innerWidth > 800 ? 280 : 'auto' },
  navActionBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, backgroundColor: '#F8FAFC' },
  navActionText: { color: COLORS.textMuted, fontWeight: '700', fontSize: 13, marginLeft: 6 },
  explicitLogoutBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, backgroundColor: '#FFF1F2' },
  explicitLogoutText: { color: COLORS.pink, fontWeight: '800', fontSize: 13, marginLeft: 6 },

  mainContainer: { flex: 1, maxWidth: 1200, width: '100%', alignSelf: 'center', paddingTop: 40, paddingHorizontal: 20 },
  pageHeader: { flexDirection: Platform.OS === 'web' && window.innerWidth > 800 ? 'row' : 'column', justifyContent: 'space-between', alignItems: Platform.OS === 'web' && window.innerWidth > 800 ? 'center' : 'flex-start', marginBottom: 25, gap: 15 },
  welcomeText: { fontSize: 26, fontWeight: '900', color: COLORS.charcoal, letterSpacing: -0.5, marginBottom: 4 },
  dateText: { fontSize: 14, color: COLORS.textMuted, fontWeight: '600' },
  newWeekBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.white, paddingHorizontal: 16, paddingVertical: 12, borderRadius: 10, borderWidth: 1, borderColor: COLORS.border, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.03, shadowRadius: 5 },
  newWeekText: { color: COLORS.charcoal, fontWeight: '800', marginLeft: 8, fontSize: 13 },

  ctaCard: { flexDirection: Platform.OS === 'web' && window.innerWidth > 800 ? 'row' : 'column', justifyContent: 'space-between', alignItems: 'center', backgroundColor: COLORS.white, padding: 30, borderRadius: 20, borderWidth: 1, borderColor: COLORS.border, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.04, shadowRadius: 12, elevation: 4, marginBottom: 30, gap: 15 },
  ctaTextCol: { flex: 1 },
  ctaTitle: { fontSize: 20, fontWeight: '900', color: COLORS.charcoal, marginBottom: 6 },
  ctaSub: { fontSize: 15, color: COLORS.textMuted, lineHeight: 22, maxWidth: 600 },
  ctaBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: COLORS.teal, paddingHorizontal: 26, paddingVertical: 16, borderRadius: 14, shadowColor: COLORS.teal, shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.3, shadowRadius: 10, elevation: 5 },
  ctaBtnText: { color: COLORS.white, fontWeight: '800', fontSize: 15, letterSpacing: 0.5 },

  filterContainer: { flexDirection: 'row', backgroundColor: '#E2E8F0', padding: 4, borderRadius: 12, alignSelf: 'flex-start', marginBottom: 20 },
  filterBtn: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: 8 },
  filterBtnActive: { backgroundColor: COLORS.white, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4, elevation: 2 },
  filterText: { fontSize: 13, fontWeight: '700', color: COLORS.textMuted },
  filterTextActive: { color: COLORS.charcoal, fontWeight: '800' },

  metricsContainer: { flexDirection: Platform.OS === 'web' && window.innerWidth > 800 ? 'row' : 'column', gap: 15, marginBottom: 30 },
  metricCard: { flex: 1, backgroundColor: COLORS.white, padding: 24, borderRadius: 16, borderWidth: 1, borderColor: COLORS.border, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.02, shadowRadius: 8, elevation: 2 },
  metricTitle: { fontSize: 12, color: COLORS.textMuted, fontWeight: '800', textTransform: 'uppercase', letterSpacing: 1 },
  numberRow: { flexDirection: 'row', alignItems: 'flex-end', marginTop: 6 },
  metricBigNumber: { fontSize: 36, fontWeight: '900', letterSpacing: -1, lineHeight: 40 },
  metricMaxNumber: { fontSize: 16, color: COLORS.textMuted, fontWeight: '700', marginBottom: 4, marginLeft: 2 },
  metricSubText: { fontSize: 12, color: COLORS.textMuted, fontWeight: '500', marginTop: 6 },

  chartCard: { backgroundColor: COLORS.white, padding: 30, borderRadius: 16, borderWidth: 1, borderColor: COLORS.border, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.03, shadowRadius: 10, elevation: 3, marginBottom: 30 },
  chartTitle: { fontSize: 18, fontWeight: '900', color: COLORS.charcoal, marginBottom: 4 },
  chartSub: { fontSize: 14, color: COLORS.textMuted, fontWeight: '500' },

  modalOverlay: { flex: 1, backgroundColor: 'rgba(15, 23, 42, 0.7)', justifyContent: 'center', alignItems: 'center', padding: 20 },
  modalCard: { backgroundColor: COLORS.white, width: '100%', maxWidth: 400, borderRadius: 24, padding: 30, alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.3, shadowRadius: 20, elevation: 15 },
  modalIconBox: { width: 60, height: 60, borderRadius: 30, justifyContent: 'center', alignItems: 'center', marginBottom: 20 },
  modalTitle: { fontSize: 20, fontWeight: '900', color: COLORS.charcoal, marginBottom: 10, textAlign: 'center' },
  modalMessage: { fontSize: 14, color: '#475569', textAlign: 'center', lineHeight: 22, marginBottom: 25 },
  modalButtonGroup: { flexDirection: 'row', width: '100%', gap: 10 },
  modalBtnCancel: { flex: 1, backgroundColor: '#F1F5F9', paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  modalBtnCancelText: { color: COLORS.charcoal, fontWeight: '800', fontSize: 14 },
  modalBtnConfirm: { flex: 1, backgroundColor: COLORS.charcoal, paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  modalBtnConfirmText: { color: COLORS.white, fontWeight: '800', fontSize: 14 },
});