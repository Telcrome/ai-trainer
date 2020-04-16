import { Component, OnInit, OnDestroy } from '@angular/core';
import { Log } from '../log';
import { CellComponent } from '../cell/cell.component';
import { DebugService } from 'src/app/debug.service';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';


@Component({
  selector: 'app-log',
  templateUrl: './log.component.html',
  styleUrls: ['./log.component.scss']
})
export class LogComponent implements OnInit, OnDestroy {

  constructor(private debugService: DebugService) { }

  logList: Observable<any[]>;

  // addLog = (log: any) => {
  //   this.logList.push(log);
  // }

  ngOnInit(): void {
    this.logList = this.debugService.getLogs();

    // this.debugService.registerCallback(data => this.logList.push({time: new Date()}));
    // let mapper = map((logjsons: Array<any>) => logjsons.forEach(x => new Log(x)));
    // this.logservice.getLogs().subscribe((x: any[]) => this.logList = Log.fromJson(x));
  }
  ngOnDestroy(): void {
    console.log('Destroyed log component');
  }

}
