import React, { useState, useEffect } from 'react'; 
import { Platform, Alert } from 'react-native';
import * as Font from 'expo-font';
import { Feather, MaterialIcons } from '@expo/vector-icons';

// --- FIREBASE IMPORTS ---
import { initializeApp } from "firebase/app";
import { getFirestore, collection, addDoc, getDocs, query, orderBy, limit } from "firebase/firestore";
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { getAnalytics } from "firebase/analytics";

// --- YOUR FIREBASE CONFIG ---
// Replace the values below with the ones from your Firebase Console
const firebaseConfig = {
  apiKey: "AIzaSyAFKSSW5tI-qnZz4I_ht5afIFYKHrkEdOM",
  authDomain: "checkgrade-by-adt.firebaseapp.com",
  projectId: "checkgrade-by-adt",
  storageBucket: "checkgrade-by-adt.firebasestorage.app",
  messagingSenderId: "44213462502",
  appId: "1:44213462502:web:5d98de4c9aafef4a1da227",
  measurementId: "G-0FFL462BG2"
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore(firebaseApp);
const storage = getStorage(firebaseApp);
if (typeof window !== 'undefined') {
  const analytics = getAnalytics(firebaseApp);
}

// --- COMPONENTS ---
import Login from './Login';
import Dashboard from './Dashboard';
import ZonesOverview from './ZonesOverview'; 
import HistoryPage from './HistoryPage'; 
import LeaderboardPage from './LeaderboardPage'; // <-- NEW: Import Leaderboard

// --- ZONES ---
import Zone1 from './CheckGrade/Zone1';   import Zone2 from './CheckGrade/Zone2';   import Zone3 from './CheckGrade/Zone3';
import Zone4 from './CheckGrade/Zone4';   import Zone5 from './CheckGrade/Zone5';   import Zone6 from './CheckGrade/Zone6';
import Zone7 from './CheckGrade/Zone7';   import Zone8 from './CheckGrade/Zone8';   import Zone9 from './CheckGrade/Zone9';
import Zone10 from './CheckGrade/Zone10'; import Zone11 from './CheckGrade/Zone11'; import Zone12 from './CheckGrade/Zone12';
import Zone13 from './CheckGrade/Zone13'; import Zone14 from './CheckGrade/Zone14'; import Zone15 from './CheckGrade/Zone15';
import Zone16 from './CheckGrade/Zone16'; import Zone17 from './CheckGrade/Zone17'; import Zone18 from './CheckGrade/Zone18';
import Zone19 from './CheckGrade/Zone19'; import Zone20 from './CheckGrade/Zone20'; import Zone21 from './CheckGrade/Zone21';
import Zone22 from './CheckGrade/Zone22'; import Zone23 from './CheckGrade/Zone23'; import Zone24 from './CheckGrade/Zone24';
import Zone25 from './CheckGrade/Zone25'; import Zone26 from './CheckGrade/Zone26'; import Zone27 from './CheckGrade/Zone27';
import Zone28 from './CheckGrade/Zone28'; import Zone29 from './CheckGrade/Zone29'; import Zone30 from './CheckGrade/Zone30';
import Zone31 from './CheckGrade/Zone31'; import Zone32 from './CheckGrade/Zone32'; import Zone33 from './CheckGrade/Zone33';
import Zone34 from './CheckGrade/Zone34'; import Zone35 from './CheckGrade/Zone35';

const ZONES = [
  { id: '1', name: 'Zone 1', badge: 'SECURITY', image: require('./assets/Zone 1.png') }, 
  { id: '2', name: 'Zone 2', badge: 'LOWER OFFICE', image: require('./assets/Zone 2.png') },
  { id: '3', name: 'Zone 3', badge: 'CRÈCHE / MEDICAL', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '4', name: 'Zone 4', badge: 'PREPATORY AREA', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '5', name: 'Zone 5', badge: 'CUTTING', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '6', name: 'Zone 6', badge: 'LINE 9', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '7', name: 'Zone 7', badge: 'CAD', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '8', name: 'Zone 8', badge: 'PRE PRODUCTION', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '9', name: 'Zone 9', badge: 'WAREHOUSE', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '10', name: 'Zone 10', badge: 'UTILITY AREA', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '11', name: 'Zone 11', badge: 'BACKSIDE AREA', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '12', name: 'Zone 12', badge: 'UPPER OFFICE', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '13', name: 'Zone 13', badge: 'SEWING FACTORY 1', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '14', name: 'Zone 14', badge: 'SEWING FACTORY 2', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '15', name: 'Zone 15', badge: 'FINISHING FACTORY 1', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '16', name: 'Zone 16', badge: 'FINISHING FACTORY 2', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '17', name: 'Zone 17', badge: 'BAGGING 1ST FL', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '18', name: 'Zone 18', badge: 'MAINTENANCE', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '19', name: 'Zone 19', badge: 'FG WAREHOUSE 1ST FL', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '20', name: 'Zone 20', badge: 'TRIMS AREA 1ST FL', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '21', name: 'Zone 21', badge: 'CSO 1ST FLOOR', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '22', name: 'Zone 22', badge: 'TRAINING', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '23', name: 'Zone 23', badge: 'SEWING FACTORY 3', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '24', name: 'Zone 24', badge: 'SEWING FACTORY 4', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '25', name: 'Zone 25', badge: 'FINISHING FACTORY 3', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '26', name: 'Zone 26', badge: 'FINISHING FACTORY 4', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '27', name: 'Zone 27', badge: 'BAGGING 2ND FL', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '28', name: 'Zone 28', badge: 'RE-BUTTONING', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '29', name: 'Zone 29', badge: 'FG WAREHOUSE 2ND FL', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '30', name: 'Zone 30', badge: 'TRIMS AREA 2ND FL', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '31', name: 'Zone 31', badge: 'CSO 2ND FLOOR', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '32', name: 'Zone 32', badge: 'MACHAN', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '33', name: 'Zone 33', badge: 'CANTEEN', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '34', name: 'Zone 34', badge: 'SEWING FACTORY 5', image: require('./assets/Zone1_Cleaning.jpeg') },
  { id: '35', name: 'Zone 35', badge: 'FINISHING FACTORY 5', image: require('./assets/Zone1_Cleaning.jpeg') },
];

const ZONE_COMPONENTS = {
  '1': Zone1, '2': Zone2, '3': Zone3, '4': Zone4, '5': Zone5,
  '6': Zone6, '7': Zone7, '8': Zone8, '9': Zone9, '10': Zone10,
  '11': Zone11, '12': Zone12, '13': Zone13, '14': Zone14, '15': Zone15,
  '16': Zone16, '17': Zone17, '18': Zone18, '19': Zone19, '20': Zone20,
  '21': Zone21, '22': Zone22, '23': Zone23, '24': Zone24, '25': Zone25,
  '26': Zone26, '27': Zone27, '28': Zone28, '29': Zone29, '30': Zone30,
  '31': Zone31, '32': Zone32, '33': Zone33, '34': Zone34, '35': Zone35,
};

export default function App() {
  const [fontsLoaded] = Font.useFonts({
    ...Feather.font,
    ...MaterialIcons.font,
  });

  const [userEmail, setUserEmail] = useState(null); 
  const [currentView, setCurrentView] = useState('dashboard'); // 'dashboard', 'zones', 'history', 'leaderboard'
  const [selectedZoneID, setSelectedZoneID] = useState(null); 
  const [auditResults, setAuditResults] = useState({}); 
  const [auditHistory, setAuditHistory] = useState([]); 

  // --- 1. Fetch Cloud Data on Load ---
  useEffect(() => {
    if (Platform.OS === 'web') {
      const savedEmail = window.localStorage.getItem('checkgrade_auditor');
      if (savedEmail) setUserEmail(savedEmail);

      const savedResults = window.localStorage.getItem('checkgrade_current_week');
      if (savedResults) setAuditResults(JSON.parse(savedResults));
    }

    const loadCloudHistory = async () => {
        try {
            console.log("Fetching vault history from Firebase...");
            const q = query(collection(db, "audit_vault"), orderBy("timestamp", "desc"), limit(50));
            const querySnapshot = await getDocs(q);
            const history = [];
            querySnapshot.forEach((doc) => {
                history.push({ id: doc.id, ...doc.data() });
            });
            setAuditHistory(history);
        } catch (e) {
            console.log("Error fetching history from Firebase:", e);
        }
    };

    if (userEmail || Platform.OS === 'web' && window.localStorage.getItem('checkgrade_auditor')) {
      loadCloudHistory();
    }
  }, [userEmail]);

  const handleLogin = (email) => {
    const finalEmail = email || 'Auditor';
    setUserEmail(finalEmail);
    if (Platform.OS === 'web') window.localStorage.setItem('checkgrade_auditor', finalEmail);
  };

  const handleLogout = () => {
    setUserEmail(null);
    setCurrentView('dashboard');
    if (Platform.OS === 'web') window.localStorage.removeItem('checkgrade_auditor');
  };

  // --- 2. Save Audit to Cloud ---
  const handleZoneComplete = async (zoneId, finalScore, detailedData, auditorName) => {
    const auditDate = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }); 
    const auditTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const zoneInfo = ZONES.find(z => z.id === zoneId);

    console.log(`☁️ Starting Cloud Sync for ${zoneInfo.name}...`);

    try {
      const updatedCheckpoints = { ...detailedData };

      for (const cpId in updatedCheckpoints) {
        const cp = updatedCheckpoints[cpId];
        
        if (cp.photoUri && cp.photoUri.startsWith('blob:')) {
          const response = await fetch(cp.photoUri);
          const blob = await response.blob();
          
          const filename = `audits/${Date.now()}_${zoneId}_${cpId}.jpg`;
          const storageRef = ref(storage, filename);
          
          console.log(`📤 Uploading image for checkpoint ${cpId}...`);
          await uploadBytes(storageRef, blob);
          
          const permanentUrl = await getDownloadURL(storageRef);
          updatedCheckpoints[cpId].photoUri = permanentUrl;
        }
      }

      const cloudRecord = {
          zoneId: zoneId,
          zoneName: zoneInfo.name,
          badge: zoneInfo.badge,
          score: finalScore,
          timestamp: Date.now(),
          date: `${auditDate} at ${auditTime}`,
          auditor: auditorName || 'Auditor',
          checkpoints: updatedCheckpoints 
      };

      console.log("💾 Saving report to Firestore...");
      const docRef = await addDoc(collection(db, "audit_vault"), cloudRecord);
      
      setAuditHistory(prev => [{ id: docRef.id, ...cloudRecord }, ...prev]);
      
      const updatedResults = { ...auditResults, [zoneId]: { score: finalScore, date: auditDate } };
      setAuditResults(updatedResults);
      if (Platform.OS === 'web') window.localStorage.setItem('checkgrade_current_week', JSON.stringify(updatedResults));

      setSelectedZoneID(null);
      console.log("✅ CLOUD SYNC COMPLETE!");

    } catch (error) {
      console.error("❌ Firebase Sync Failed:", error);
      if (Platform.OS === 'web') {
          alert("Failed to save audit to cloud. Check your connection or Firebase config.");
      } else {
          Alert.alert("Cloud Sync Error", "The audit was completed but couldn't be saved to the cloud.");
      }
      setSelectedZoneID(null); 
    }
  };

  const confirmNewWeek = () => {
    setAuditResults({});
    if (Platform.OS === 'web') window.localStorage.removeItem('checkgrade_current_week');
  };

  // --- ROUTER LOGIC ---
  if (!fontsLoaded) {
    return null;
  }

  if (!userEmail) {
    return <Login onLogin={handleLogin} />;
  }

  // Zone Audit View
  if (selectedZoneID) {
    const ActiveZoneComponent = ZONE_COMPONENTS[selectedZoneID];
    if (ActiveZoneComponent) return (
      <ActiveZoneComponent 
        goBack={() => setSelectedZoneID(null)} 
        onComplete={(score, details, name) => handleZoneComplete(selectedZoneID, score, details, name)} 
        auditHistory={auditHistory} // <--- ADD THIS NEW LINE!
      />
    );
  }

  // Leaderboard View
  if (currentView === 'leaderboard') {
    return (
      <LeaderboardPage 
        historyData={auditHistory} 
        onNavigate={setCurrentView}
        onLogout={handleLogout}
      />
    );
  }

  // History Vault View
  if (currentView === 'history') {
    return (
      <HistoryPage 
        userEmail={userEmail}
        historyData={auditHistory} 
        onNavigate={setCurrentView}
        onLogout={handleLogout}
      />
    );
  }

  // Zones Grid View
  if (currentView === 'zones') {
    return (
      <ZonesOverview 
        userEmail={userEmail}
        zones={ZONES}
        auditResults={auditResults}
        auditHistory={auditHistory} // <--- ADDED THIS LINE HERE!
        onSelectZone={setSelectedZoneID}
        onNavigate={setCurrentView}
        onLogout={handleLogout}
      />
    );
  }

  // Dashboard Main View
  return (
    <Dashboard 
      userEmail={userEmail}
      zones={ZONES}
      auditResults={auditResults}
      auditHistory={auditHistory} 
      onNavigate={setCurrentView}
      onLogout={handleLogout}
      onArchiveWeek={confirmNewWeek}
    />
  );
}