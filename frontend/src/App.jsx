import React from 'react';
import { ReactFlowProvider } from 'reactflow';

import 'reactflow/dist/style.css';
import './App.css'
import MainContextProvider from './components/context/MainContext';
import DrawZone from './components/DrawZone/DrawZone';
import Sidebar from './components/SideBar/SideBar';

export default function App() {
  return (
    <MainContextProvider>
      <ReactFlowProvider>
        <div className='main'>
          <div className='header'>YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO</div>
          <div className='work-area'>
            <Sidebar />
            <DrawZone/>
          </div>
        <footer></footer>
        </div>
      </ReactFlowProvider>
    </MainContextProvider>
  );
}