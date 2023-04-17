import React from 'react';
import { ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import './App.css'
import MainContextProvider from './components/context/MainContext';
import DrawZone from './components/DrawZone/DrawZone';
import Sidebar from './components/SideBar/SideBar';
import ButtonsArea from './components/areas/ButtonsArea/ButtonsArea';

export default function App() {
  return (
      <MainContextProvider>
      <header> test for header</header>
      <div className='main-page'>
        <Sidebar />
        <div className='content-area'>
          <ButtonsArea />
          <ReactFlowProvider>
            <DrawZone/>
          </ReactFlowProvider>
        </div>
      </div>
      </MainContextProvider>
  );
}