import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, SafeAreaView, TouchableOpacity, ScrollView, Platform, StatusBar, Image } from 'react-native';
import { Feather } from '@expo/vector-icons';

const COLORS = {
  charcoal: '#0F172A', teal: '#0D9488', pink: '#E11D48', 
  bg: '#F8FAFC', white: '#FFFFFF', border: '#E2E8F0',
  textMuted: '#64748B', orange: '#F59E0B', green: '#10B981', gold: '#FFD700'
};

export default function LeaderboardPage({ historyData, onNavigate, onLogout }) {
  const [rankings, setRankings] = useState([]);
  const isMobile = Platform.OS === 'web' ? window.innerWidth <= 600 : false;

  useEffect(() => {
    calculateRankings();
  }, [historyData]);

  const calculateRankings = () => {
    const deptStats = {};

    // Group scores by Department (Badge)
    historyData.forEach(item => {
      const dept = item.badge || 'Other';
      if (!deptStats[dept]) {
        deptStats[dept] = { name: dept, totalScore: 0, count: 0 };
      }
      deptStats[dept].totalScore += parseFloat(item.score);
      deptStats[dept].count += 1;
    });

    // Calculate Averages and Sort
    const sorted = Object.values(deptStats).map(d => ({
      name: d.name,
      avg: (d.totalScore / d.count).toFixed(2)
    })).sort((a, b) => b.avg - a.avg);

    setRankings(sorted);
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor={COLORS.white} />
      
      {/* GLOBAL TOP NAV */}
      <View style={styles.topNav}>
        <View style={styles.navLeft}>
            <View style={styles.logoBox}>
                <Image source={require('./assets/logo.png')} style={styles.logoImage} />
            </View>
        </View>

        <View style={styles.navCenter}>
            <TouchableOpacity style={styles.navMenuBtn} onPress={() => onNavigate('dashboard')}>
                <Feather name="pie-chart" size={16} color={COLORS.textMuted} />
                {!isMobile && <Text style={styles.navMenuText}>Dashboard</Text>}
            </TouchableOpacity>
            
            {/* Active Tab */}
            <TouchableOpacity style={[styles.navMenuBtn, styles.navMenuBtnActive]} disabled>
                <Feather name="award" size={16} color={COLORS.teal} />
                {!isMobile && <Text style={[styles.navMenuText, styles.navMenuTextActive]}>Leaderboard</Text>}
            </TouchableOpacity>

            <TouchableOpacity style={styles.navMenuBtn} onPress={() => onNavigate('history')}>
                <Feather name="archive" size={16} color={COLORS.textMuted} />
                {!isMobile && <Text style={styles.navMenuText}>Vault</Text>}
            </TouchableOpacity>
        </View>

        <View style={styles.navRight}>
            <TouchableOpacity style={styles.explicitLogoutBtn} onPress={onLogout}>
                <Feather name="log-out" size={16} color={COLORS.pink} />
            </TouchableOpacity>
        </View>
      </View>

      <ScrollView style={styles.mainContainer} showsVerticalScrollIndicator={false}>
        <View style={styles.pageHeader}>
            <Text style={styles.welcomeText}>Factory Rankings</Text>
            <Text style={styles.dateText}>Real-time 5S compliance performance by department</Text>
        </View>

        {rankings.length > 0 ? (
          <View>
            {/* PODIUM SECTION */}
            <View style={styles.podiumContainer}>
                {/* 2nd Place */}
                {rankings[1] && (
                    <View style={[styles.podiumItem, { height: 140 }]}>
                        <Feather name="award" size={30} color="#94A3B8" />
                        <Text style={styles.podiumName}>{rankings[1].name}</Text>
                        <Text style={styles.podiumScore}>{rankings[1].avg}</Text>
                    </View>
                )}
                {/* 1st Place */}
                {rankings[0] && (
                    <View style={[styles.podiumItem, styles.podiumFirst, { height: 180 }]}>
                        <Feather name="star" size={40} color={COLORS.gold} />
                        <Text style={[styles.podiumName, { fontSize: 18 }]}>{rankings[0].name}</Text>
                        <Text style={[styles.podiumScore, { fontSize: 28 }]}>{rankings[0].avg}</Text>
                    </View>
                )}
                {/* 3rd Place */}
                {rankings[2] && (
                    <View style={[styles.podiumItem, { height: 120 }]}>
                        <Feather name="award" size={24} color="#B45309" />
                        <Text style={styles.podiumName}>{rankings[2].name}</Text>
                        <Text style={styles.podiumScore}>{rankings[2].avg}</Text>
                    </View>
                )}
            </View>

            {/* FULL LIST */}
            <View style={styles.listCard}>
                {rankings.map((rank, index) => (
                    <View key={rank.name} style={[styles.rankRow, index === rankings.length - 1 && { borderBottomWidth: 0 }]}>
                        <View style={styles.rankLeft}>
                            <Text style={styles.rankIndex}>{index + 1}</Text>
                            <Text style={styles.rankName}>{rank.name}</Text>
                        </View>
                        <View style={[styles.rankBadge, { backgroundColor: parseFloat(rank.avg) >= 4 ? '#ECFDF5' : '#FFFBEB' }]}>
                            <Text style={[styles.rankAvg, { color: parseFloat(rank.avg) >= 4 ? COLORS.green : COLORS.orange }]}>{rank.avg}</Text>
                        </View>
                    </View>
                ))}
            </View>
          </View>
        ) : (
          <View style={styles.emptyState}>
              <Feather name="bar-chart-2" size={60} color="#CBD5E1" />
              <Text style={styles.emptyText}>Complete audits to see rankings</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: COLORS.bg },
  topNav: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: COLORS.white, paddingHorizontal: 20, height: 100, borderBottomWidth: 1, borderBottomColor: COLORS.border, zIndex: 10 },
  logoBox: { width: 180, height: 60, justifyContent: 'center', alignItems: 'flex-start' },
  logoImage: { width: '100%', height: '100%', resizeMode: 'contain' },
  navCenter: { flexDirection: 'row', gap: 10, flex: 1, justifyContent: 'center' },
  navMenuBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 10, borderRadius: 12 },
  navMenuBtnActive: { backgroundColor: '#F0FDFA' },
  navMenuText: { color: COLORS.textMuted, fontWeight: '700', fontSize: 14, marginLeft: 8 },
  navMenuTextActive: { color: COLORS.teal },
  navRight: { width: 200, alignItems: 'flex-end' },
  explicitLogoutBtn: { padding: 12, borderRadius: 8, backgroundColor: '#FFF1F2' },
  
  mainContainer: { flex: 1, maxWidth: 1000, width: '100%', alignSelf: 'center', paddingTop: 40, paddingHorizontal: 20 },
  pageHeader: { marginBottom: 40, alignItems: 'center' },
  welcomeText: { fontSize: 32, fontWeight: '900', color: COLORS.charcoal, marginBottom: 8 },
  dateText: { fontSize: 16, color: COLORS.textMuted, fontWeight: '600', textAlign: 'center' },

  podiumContainer: { flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'center', gap: 15, marginBottom: 40 },
  podiumItem: { flex: 1, backgroundColor: COLORS.white, borderRadius: 20, padding: 20, alignItems: 'center', justifyContent: 'center', borderWidth: 1, borderColor: COLORS.border, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 5 },
  podiumFirst: { borderColor: COLORS.gold, borderWidth: 2, shadowColor: COLORS.gold, shadowOpacity: 0.15 },
  podiumName: { fontWeight: '800', color: COLORS.charcoal, marginTop: 10, textAlign: 'center' },
  podiumScore: { fontSize: 22, fontWeight: '900', color: COLORS.teal, marginTop: 5 },

  listCard: { backgroundColor: COLORS.white, borderRadius: 24, padding: 10, borderWidth: 1, borderColor: COLORS.border, marginBottom: 100 },
  rankRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 20, borderBottomWidth: 1, borderBottomColor: '#F1F5F9' },
  rankLeft: { flexDirection: 'row', alignItems: 'center' },
  rankIndex: { fontSize: 18, fontWeight: '900', color: COLORS.textMuted, width: 40 },
  rankName: { fontSize: 18, fontWeight: '800', color: COLORS.charcoal },
  rankBadge: { paddingHorizontal: 15, paddingVertical: 8, borderRadius: 10 },
  rankAvg: { fontSize: 16, fontWeight: '900' },

  emptyState: { alignItems: 'center', marginTop: 100 },
  emptyText: { color: COLORS.textMuted, fontSize: 18, fontWeight: '600', marginTop: 20 }
});