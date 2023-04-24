import { useEffect } from 'react';
import 'overlayscrollbars/overlayscrollbars.css';
import { OverlayScrollbars } from 'overlayscrollbars';

const useScrollbar = (root, hasScroll) => {
  useEffect(() => {
    let scrollbars;

    if ((root.current, hasScroll)) {
      scrollbars = OverlayScrollbars(root.current, {});
    }

    return () => {
      if (scrollbars) {
        scrollbars.destroy();
      }
    };
  }, [root, hasScroll]);
};

export { useScrollbar };
