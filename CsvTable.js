import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, ScrollView } from "react-native";
import Papa from "papaparse";

const BACKEND_URL = "http://10.0.2.2:10000/stats-csv";
// Example: http://192.168.1.10:3000/stats-csv

export default function CsvTable() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch(BACKEND_URL)
      .then((res) => res.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            setData(results.data.slice(0, 10)); // first 10 rows
          },
        });
      })
      .catch((err) => console.error("CSV fetch error:", err));
  }, []);

  if (!data.length) {
    return <Text style={styles.loading}>Loading stats.csv…</Text>;
  }

  const headers = Object.keys(data[0]);

  return (
    <ScrollView horizontal style={styles.wrapper}>
      <View>
        <View style={[styles.row, styles.header]}>
          {headers.map((h) => (
            <Text key={h} style={[styles.cell, styles.headerText]}>
              {h}
            </Text>
          ))}
        </View>

        {data.map((row, i) => (
          <View key={i} style={styles.row}>
            {headers.map((h) => (
              <Text key={h} style={styles.cell}>{row[h]}</Text>
            ))}
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  wrapper: { marginTop: 16 },
  loading: { marginTop: 12, color: "#666" },
  row: { flexDirection: "row" },
  header: { backgroundColor: "#eee" },
  cell: {
    minWidth: 110,
    padding: 6,
    fontSize: 13,
    borderWidth: 1,
    borderColor: "#ccc",
  },
  headerText: { fontWeight: "600" },
});
