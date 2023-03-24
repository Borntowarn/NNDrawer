import React from 'react';
import { ReactFlowProvider } from 'reactflow';

import 'reactflow/dist/style.css';
import './App.css'
import MainContextProvider from './components/context/MainContext';
import DrawZone from './components/DrawZone/DrawZone';

export default function App() {
  return (
    <MainContextProvider>
      <ReactFlowProvider>
        <DrawZone/>
      </ReactFlowProvider>
    </MainContextProvider>
  );
}