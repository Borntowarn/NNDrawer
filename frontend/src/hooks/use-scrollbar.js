import { useEffect } from 'react';
import 'overlayscrollbars/overlayscrollbars.css';
import { OverlayScrollbars } from 'overlayscrollbars';

const useScrollbar = (root, hasScroll) => {
  useEffect(() => {
    console.log(root, hasScroll);
    let scrollbars;

    console.log('CHANGE_ROOT:', root, hasScroll);

    if ((root.current, hasScroll)) {
      scrollbars = OverlayScrollbars(root.current, {});
    }

    console.log('SCROLLBARS: ', scrollbars);

    return () => {
      if (scrollbars) {
        scrollbars.destroy();
      }
    };
  }, [root, hasScroll]);
};

export { useScrollbar };
