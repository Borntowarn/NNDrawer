import { ReactFlowProvider } from 'reactflow';
import MainContextProvider from './components/context/MainContext';
import DrawZone from './components/DrawZone/DrawZone';
import Sidebar from './components/SideBar/SideBar';
import ButtonsArea from './components/areas/ButtonsArea/ButtonsArea';
import AuthorizationModal from './components/modals/AuthorizationModal/AuthorizationModal';
import RegistrationModal from './components/modals/RegistrationModal/RegistrationModal';
import Header from './components/Header/Header';
import 'reactflow/dist/style.css';
import './App.css'

export default function App() {
  return (
      <MainContextProvider>
      <Header />
      <div className='main-page'>
        <Sidebar />
        <div className='content-area'>
          <ReactFlowProvider>
            <ButtonsArea />
            <DrawZone/>
          </ReactFlowProvider>
        </div>
      </div>
      <AuthorizationModal />
      {/* <RegistrationModal /> */}
      </MainContextProvider>
  );
}